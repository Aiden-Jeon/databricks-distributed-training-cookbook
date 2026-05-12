# TorchDistributor 내부 동작

TorchDistributor가 Spark 위에서 정확히 어떻게 child 프로세스를 띄우고 rendezvous를 잡는지 정리합니다. 분산 학습 자체는 익숙하지만 "Databricks/Spark layer가 어디까지 개입하는지" 가 black box처럼 느껴질 때 참고합니다.

## 한 줄 정리

| `local_mode` | 프로세스 launch 주체 | child 위치 | rendezvous |
|--------------|--------------------|-----------|------------|
| `True` | driver 노드 안의 Python `subprocess` (Spark task 미사용) | driver 머신 N개 프로세스 | localhost (`MASTER_ADDR=127.0.0.1`) |
| `False` | Spark **barrier execution mode** task (`SparkContext.runJob`) | worker 노드 executor 위에 task별로 1개 child | rank 0가 잡힌 worker의 IP가 `MASTER_ADDR` |

핵심: `local_mode=False` 일 때 child는 **Spark task로 도는 Python 프로세스**입니다. executor JVM이 forkserver를 통해 child Python을 띄우고, 그 안에서 학습 함수가 실행됩니다. Spark의 일반 task와 다른 점은 **barrier mode** 라는 것 — 모든 task가 rendezvous 시점에 동시에 시작되도록 보장하는 mode입니다 (Spark 2.4+의 gang scheduling).

## launcher 흐름 (`local_mode=False`)

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

driver의 역할은 코디네이션과 결과 회수뿐입니다. driver에 GPU가 없어도 동작하지만, `local_mode=True` 와 클러스터를 공유한다면 GPU 인스턴스로 둡니다 (driver 학습 시 그 GPU를 사용).

## launcher 흐름 (`local_mode=True`)

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

`local_mode=True` 는 **Spark을 거의 쓰지 않습니다**. SparkContext는 keepalive 용도로 살아 있지만 task는 submit되지 않습니다. 그래서 single-node 클러스터(worker=0)에서도 동작합니다 — driver만 있어도 됩니다.

## RANK / WORLD_SIZE / MASTER_ADDR 세팅

DDP 경험자가 torchrun에서 보던 환경변수와 동일합니다. TorchDistributor가 자동으로 채워 주는 값:

| 변수 | `local_mode=True` | `local_mode=False` |
|------|-------------------|---------------------|
| `RANK` | 0..N-1 (driver 내 subprocess index) | 0..M*N-1 (Spark task index) |
| `LOCAL_RANK` | RANK와 동일 | task가 잡힌 노드 내에서의 GPU 인덱스 |
| `WORLD_SIZE` | N | M*N |
| `MASTER_ADDR` | 127.0.0.1 | rank 0 task가 스케줄된 worker의 host |
| `MASTER_PORT` | 자동 할당 (29500 기본) | 자동 할당 |

`debug-common-pitfalls.md` §1 "직접 만들지 말 것" 의 의미: 위 값들이 모두 **TorchDistributor가 task 배치 시점에 계산** 합니다. 사용자가 노트북에서 `os.environ["MASTER_ADDR"]=...` 로 미리 잡으면 task별 정확한 값을 덮어쓰게 되어 NCCL rendezvous가 깨집니다.

## NCCL rendezvous 단계

`dist.init_process_group("nccl")` 가 child에서 호출되면:

1. 모든 rank가 `MASTER_ADDR:MASTER_PORT` 로 TCP store에 접속
2. rank 0가 NCCL `UniqueId` 를 생성해 store에 publish
3. 다른 rank들이 `UniqueId` 를 받아 NCCL communicator 초기화
4. 모든 GPU 간 NCCL 링이 형성되면 통과

이 단계에서 멈춘다면 보통 **방화벽**(MASTER_PORT 차단), **노드 간 IB/EFA 미활성** (default ethernet으로 fallback 후 timeout), 또는 **GPU driver 버전 mismatch** 입니다.

## 왜 `TorchDistributor.run` 을 연속 호출하면 py4j callback이 단절되는가

`02-script-based/README.md:56` 와 `03-custom-package-script-based/README.md:20` 에서 언급된 현상(DBR 17.3 LTS ML, g5.12xlarge 관찰). 근본 원인은 확정되지 않았지만 가능한 가설:

1. **TorchDistributor가 cloudpickle 직렬화 시 driver-side `SparkContext`/`SparkSession` 참조를 캡처할 수 있음.** child가 그 참조를 deserialize하려고 py4j gateway에 접근하면, 두 번째 호출 시점에 gateway가 stale 상태일 수 있습니다.
2. **barrier execution mode가 SparkContext의 `BarrierTaskContext` 상태를 변경**합니다. 연속 호출 시 이전 barrier job의 정리가 늦어지면서 callback channel이 일시적으로 끊길 수 있습니다.
3. **child 프로세스가 종료 시 driver-side py4j callback 서버에 등록된 listener를 정리하지 못하는 경우**가 있어, 누적된 stale callback이 다음 호출의 새 callback과 충돌.

증상: 두 번째 `TorchDistributor.run` 호출은 시작되지만 학습 중간에 `Py4JError: An error occurred while calling ... callbackServer` 같은 에러로 죽거나, 학습은 끝나는데 driver에서 결과 수집이 안 됨.

회피책 (본 쿡북 채택): **launcher × topology 별로 노트북을 분리**해서 매번 fresh Python interpreter로 시작. 같은 노트북에서 `dbutils.notebook.exit` → 재실행도 가능하지만, 노트북 분리가 더 깔끔합니다.

다른 회피책 (참고):
- `spark.conf.set("spark.databricks.python.barrier.enabled", "true")` 명시 (이미 default).
- 두 호출 사이에 `SparkContext._gateway.callback_server.shutdown()` 후 재시작 — fragile해서 권장하지 않음.

## driver의 GPU는 언제 쓰이는가

| 시나리오 | driver GPU 사용 |
|---------|-----------------|
| `local_mode=True, num_processes=N` | ✅ driver의 N개 GPU |
| `local_mode=False, num_processes=M*N` | ❌ driver는 코디네이션만 |
| Lightning `Trainer.fit()` 직접 호출 (no TorchDistributor) | ✅ driver의 GPU |

`local_mode=False` 에서 driver를 CPU 인스턴스로 두는 것이 비용상 합리적이지만, **같은 클러스터에서 `local_mode=True` 노트북도 실행할 거면 driver를 GPU로** 둬야 합니다. 본 쿡북은 02·04 (single-node)와 03·05 (multi-node) 노트북을 **별도 클러스터**로 분리하는 것을 권장 ([`01-notebook-based/README.md`](../01-notebook-based/README.md)).

## use_gpu / use_cuda 인자

`TorchDistributor(..., use_gpu=True)` 는 child 프로세스에 `CUDA_VISIBLE_DEVICES` 를 자동 세팅합니다.

- `local_mode=True`: driver 머신의 GPU 인덱스 0..N-1 을 각 child에 1개씩 노출
- `local_mode=False`: 각 worker executor가 할당받은 GPU를 그 노드 내 child에 노출

`use_gpu=False` 로 두면 NCCL이 아닌 gloo backend를 가정한 것으로 처리됩니다. 본 쿡북은 항상 `use_gpu=True`.

## Spark UI에서 확인

`local_mode=False` 학습 중에 Spark UI → Jobs 탭을 보면 **barrier stage** 하나가 떠 있습니다. task 수 = `M*N`. 각 task의 stderr가 executor log로 흘러 들어갑니다 — worker 노드의 NCCL/PyTorch 로그를 보려면 거기서 확인 (`debug-common-pitfalls.md` 의 "Worker 로그 읽는 법" 섹션 참고).

## 한계와 대안

| 한계 | 대안 |
|------|------|
| 같은 노트북에서 `.run()` 연속 호출 불안정 | 노트북 분리 (본 쿡북 패턴) |
| child가 driver 자격증명 자동 상속 안 함 | host/token 명시 전달 ([debug-common-pitfalls.md §2-1](debug-common-pitfalls.md)) |
| cloudpickle 직렬화 한계 (SparkSession 등) | `train_fn` 모듈 최상위 정의, 인자는 primitives만 ([debug-common-pitfalls.md §2](debug-common-pitfalls.md)) |
| Autoscaling과 호환 안 됨 | DDP는 시작 시 노드 수 고정 → autoscaling OFF |
| 학습 중 노드 교체 (spot) 불가 | spot interruption 시 전체 재시작 — [`ops-resume-training.md`](ops-resume-training.md) |

## 참고

- TorchDistributor 소스 (PySpark 3.5+): https://github.com/apache/spark/blob/master/python/pyspark/ml/torch/distributor.py
- Spark Barrier Execution Mode (SPARK-24374): https://issues.apache.org/jira/browse/SPARK-24374
- Databricks Blog – Introducing TorchDistributor: https://www.databricks.com/blog/2023/04/20/pytorch-databricks-introducing-spark-pytorch-distributor.html
- PyTorch DDP rendezvous: https://pytorch.org/docs/stable/distributed.html#tcp-initialization
