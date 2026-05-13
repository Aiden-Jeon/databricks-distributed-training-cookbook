# TorchDistributor 내부 동작

TorchDistributor가 Spark 위에서 어떻게 child 프로세스를 띄우고 rendezvous를 잡는지 정리합니다. 분산 학습 자체는 익숙하지만 "Databricks/Spark layer가 어디까지 개입하는지"가 black box처럼 느껴질 때 참고하면 됩니다.

## `local_mode` 플래그

TorchDistributor에서 가장 먼저 부딪히는 설정이 `local_mode` 플래그입니다. 이름이 다소 모호하지만 실제로는 **분산 학습 프로세스를 어디에 띄울 것인가** — driver 노드 안인지, Spark worker 노드들에 분산할 것인지를 결정합니다.

- `local_mode=True`는 "driver에서 학습한다"는 뜻으로 읽으면 됩니다. Single-node 클러스터(driver만 있고 worker=0)나 driver에 GPU가 붙어 있는 클러스터에서 사용합니다. TorchDistributor가 driver 머신 안에서 `num_processes`개의 Python 프로세스를 띄우고 NCCL로 묶어 줍니다.
- `local_mode=False`는 반대로 "worker에서 학습한다"는 뜻입니다. Spark가 worker 노드들에 child 프로세스를 분배하고, driver는 학습 자체에는 끼지 않고 RPC 코디네이션만 담당합니다. 이 차이는 모니터링에도 그대로 영향을 주는데, multi-node에서 driver-side `log_system_metrics=True`만 켜면 driver는 학습을 하지 않기 때문에 idle 메트릭만 기록됩니다 ([`ops-mlflow-tracking.md §1`](ops-mlflow-tracking.md)).

본 쿡북의 각 토폴로지가 위 두 mode 중 무엇에 해당하는지 정리하면 다음과 같습니다.

| 셀 | 호출 | 의미 |
|----|------|------|
| 1×1 | `TorchDistributor(num_processes=1, local_mode=True)` | driver에서 1개 프로세스. world_size=1이라 DDP all_reduce는 no-op. |
| 1×N | `TorchDistributor(num_processes=N, local_mode=True)` | driver의 N개 GPU에 N개 프로세스. |
| M×N | `TorchDistributor(num_processes=M*N, local_mode=False)` | M개 worker 노드에 총 M*N 프로세스 분배. driver는 코디네이션만. |

두 mode의 차이를 한 표로 압축하면 다음과 같습니다.

| `local_mode` | 프로세스 launch 주체 | child 위치 | rendezvous |
|--------------|--------------------|-----------|------------|
| `True` | driver 노드 안의 Python `subprocess` (Spark task 미사용) | driver 머신 N개 프로세스 | localhost (`MASTER_ADDR=127.0.0.1`) |
| `False` | Spark **barrier execution mode** task (`SparkContext.runJob`) | worker 노드 executor 위에 task별로 1개 child | rank 0가 잡힌 worker의 IP가 `MASTER_ADDR` |

핵심은 `local_mode=False`일 때 child가 **Spark task로 도는 Python 프로세스**라는 점입니다. executor JVM이 forkserver를 통해 child Python을 띄우고, 그 안에서 학습 함수가 실행됩니다. Spark의 일반 task와 다른 점은 **barrier mode**라는 것으로, 모든 task가 rendezvous 시점에 동시에 시작되도록 보장하는 mode입니다(Spark 2.4+의 gang scheduling).

이 플래그를 잘못 맞추면 학습 자체가 시작되지 않거나, 시작되더라도 의도와 전혀 다른 모양으로 돌아갑니다. 자주 마주치는 함정은 다음과 같습니다.

- Single-node 클러스터에서 `local_mode=False`로 호출하면 분배할 worker가 없어 그대로 실패합니다.
- Multi-node 클러스터에서 `local_mode=True`로 호출하면 비싼 worker 노드들이 놀고 driver 한 대에서만 학습이 진행됩니다.
- `local_mode=False`로 띄운 child 프로세스는 다른 노드의 fresh Python에서 시작하므로 driver의 자격증명이나 import 상태를 자동으로 상속받지 못합니다. 실수의 단골 원인이니 따로 살펴 두는 것이 좋습니다 ([`debug-common-pitfalls.md §2`, `§2-1`](../docs/debug-common-pitfalls.md)).

## launcher 실행 흐름

driver와 executor가 각각 무엇을 하는지 mode별로 풀어 보면 다음과 같습니다. 단순한 `local_mode=True`부터 봅니다.

### `local_mode=True`

`local_mode=True`는 Spark 경로를 거의 쓰지 않고 driver 안에서 모든 일을 끝냅니다.

```
[Driver 노트북]
  ↓ TorchDistributor(num_processes=N, local_mode=True).run(fn, **kwargs)
[Driver]
  1. cloudpickle.dumps(fn, kwargs)
  2. driver 머신 내부에서 N개 Python subprocess 직접 spawn
     (Spark task 사용 안 함)
  3. RANK/WORLD_SIZE 등을 환경변수로 주입
  4. 모든 subprocess 완료까지 대기
```

`local_mode=True`는 **Spark을 거의 쓰지 않습니다**. SparkContext는 keepalive 용도로 살아 있지만 task는 submit되지 않습니다. 덕분에 single-node 클러스터(worker=0)에서도 동작하며, driver 한 대만 있으면 됩니다.

### `local_mode=False`

`local_mode=False`는 Spark barrier execution mode를 통해 worker 노드들에 child를 분배합니다.

```
[Driver 노트북]
  ↓ TorchDistributor(...).run(fn, **kwargs)
[Driver: TorchDistributor]
  1. cloudpickle.dumps(fn, kwargs)           # by-value 직렬화
  2. M*N개 Spark task를 barrier mode로 submit
  3. 각 task 시작 시점에 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT를
     task index 기반으로 계산해 환경변수로 주입
  4. SparkContext.runJob이 block — 모든 task 완료까지 대기
  5. driver는 Spark UI와 stderr로 task 진행 모니터링만 함

[Executor (각 worker 노드)]
  - Spark task = Python child 프로세스 1개 = DDP rank 1개
  - 환경변수는 이미 세팅됨 → fn(**kwargs) 호출
  - fn 내부에서 dist.init_process_group("nccl")이 NCCL rendezvous 시작
  - 모든 rank가 init_process_group 통과해야 학습이 시작됨 (barrier)
```

driver의 역할은 코디네이션과 결과 회수뿐입니다. driver에 GPU가 없어도 동작하지만, `local_mode=True`와 클러스터를 공유한다면 GPU 인스턴스로 둡니다(driver 학습 시 그 GPU를 사용).

## 환경변수와 NCCL rendezvous

DDP 경험자가 torchrun에서 보던 환경변수와 동일합니다. TorchDistributor가 채워 주는 값은 다음과 같습니다.

| 변수 | `local_mode=True` | `local_mode=False` |
|------|-------------------|---------------------|
| `RANK` | 0..N-1 (driver 내 subprocess index) | 0..M*N-1 (Spark task index) |
| `LOCAL_RANK` | RANK와 동일 | task가 잡힌 노드 내에서의 GPU 인덱스 |
| `WORLD_SIZE` | N | M*N |
| `MASTER_ADDR` | 127.0.0.1 | rank 0 task가 스케줄된 worker의 host |
| `MASTER_PORT` | 자동 할당 (29500 기본) | 자동 할당 |

[`../docs/debug-common-pitfalls.md`](../docs/debug-common-pitfalls.md) §1 "직접 만들지 말 것"이 가리키는 지점이 바로 이 표입니다. 위 값들은 모두 **TorchDistributor가 task 배치 시점에 계산**합니다. 사용자가 노트북에서 `os.environ["MASTER_ADDR"]=...`로 미리 잡으면 task별로 계산된 정확한 값을 덮어쓰게 되어 NCCL rendezvous가 깨집니다.

`dist.init_process_group("nccl")`가 child에서 호출되면 다음 순서로 rendezvous가 진행됩니다.

1. 모든 rank가 `MASTER_ADDR:MASTER_PORT`로 TCP store에 접속
2. rank 0이 NCCL `UniqueId`를 생성해 store에 publish
3. 다른 rank들이 `UniqueId`를 받아 NCCL communicator 초기화
4. 모든 GPU 간 NCCL 링이 형성되면 통과

이 단계에서 멈추면 보통 **방화벽**(MASTER_PORT 차단), **노드 간 IB/EFA 미활성**(default ethernet으로 fallback 후 timeout), 또는 **GPU driver 버전 mismatch**가 원인입니다.

## GPU 할당

`TorchDistributor(..., use_gpu=True)`는 child 프로세스에 `CUDA_VISIBLE_DEVICES`를 자동 세팅합니다.

- `local_mode=True`: driver 머신의 GPU 인덱스 0..N-1을 각 child에 1개씩 노출
- `local_mode=False`: 각 worker executor가 할당받은 GPU를 그 노드 내 child에 노출

`use_gpu=False`로 두면 NCCL이 아닌 gloo backend를 가정한 것으로 처리됩니다. 본 쿡북은 항상 `use_gpu=True`로 둡니다.

토폴로지별로 driver GPU가 실제로 학습에 참여하는지는 다음과 같습니다.

| 시나리오 | driver GPU 사용 |
|---------|-----------------|
| `local_mode=True, num_processes=N` | O — driver의 N개 GPU |
| `local_mode=False, num_processes=M*N` | X — driver는 코디네이션만 |
| Lightning `Trainer.fit()` 직접 호출 (no TorchDistributor) | O — driver의 GPU |

`local_mode=False`만 쓴다면 driver를 CPU 인스턴스로 두는 편이 비용상 합리적이지만, **같은 클러스터에서 `local_mode=True` 노트북도 실행할 거라면 driver를 GPU로** 둬야 합니다. 본 쿡북은 02·04(single-node)와 03·05(multi-node) 노트북을 **별도 클러스터**로 분리하는 것을 권장합니다([`01-notebook-based/README.md`](../01-notebook-based/README.md)).

## 운영 관점

### Spark UI에서 확인

`local_mode=False` 학습 중에 Spark UI의 Jobs 탭을 보면 **barrier stage** 하나가 떠 있습니다. task 수는 `M*N`이고, 각 task의 stderr가 executor log로 흘러 들어갑니다. worker 노드의 NCCL/PyTorch 로그를 보려면 거기서 확인합니다([`../docs/debug-common-pitfalls.md`](../docs/debug-common-pitfalls.md)의 "Worker 로그 읽는 법" 섹션 참고).

### 한계와 대안

TorchDistributor를 쓰면서 부딪히는 주요 한계와 대처를 정리하면 다음과 같습니다.

| 한계 | 대안 |
|------|------|
| 같은 노트북에서 `.run()` 연속 호출 불안정 | 노트북 분리 (본 쿡북 패턴) — [debug-common-pitfalls.md §12](../docs/debug-common-pitfalls.md#12-같은-노트북에서-torchdistributorrun-연속-호출-시-py4j-callback-단절) |
| child가 driver 자격증명 자동 상속 안 함 | host/token 명시 전달 ([debug-common-pitfalls.md §2-1](../docs/debug-common-pitfalls.md)) |
| cloudpickle 직렬화 한계 (SparkSession 등) | `train_fn` 모듈 최상위 정의, 인자는 primitives만 ([debug-common-pitfalls.md §2](../docs/debug-common-pitfalls.md)) |
| Autoscaling과 호환 안 됨 | DDP는 시작 시 노드 수 고정 → autoscaling OFF |
| 학습 중 노드 교체 (spot) 불가 | spot interruption 시 전체 재시작 — [`ops-resume-training.md`](ops-resume-training.md) |

각 한계는 결국 "DDP는 시작 시점에 world를 고정한다"는 전제에서 파생됩니다. 그래서 대안은 모두 외부에서 시도를 다시 시작하거나, 직렬화 경계를 명확히 끊는 방향으로 잡혀 있습니다.

## 참고

자세한 내용은 다음 자료를 참조하세요.

- [TorchDistributor 소스 (PySpark 3.5+)](https://github.com/apache/spark/blob/master/python/pyspark/ml/torch/distributor.py)
- [Spark Barrier Execution Mode (SPARK-24374)](https://issues.apache.org/jira/browse/SPARK-24374)
- [Databricks Blog: Introducing TorchDistributor](https://www.databricks.com/blog/2023/04/20/pytorch-databricks-introducing-spark-pytorch-distributor.html)
- [PyTorch DDP rendezvous](https://pytorch.org/docs/stable/distributed.html#tcp-initialization)
