# 흔한 함정

이 문서는 분산 학습에서 자주 부딪히는 실패와 해결법을 모아 둔 곳입니다. 디버깅 중이라면 여기부터 살펴보면 됩니다.

## 진단 — 에러 메시지로 항목 찾기

먼저 에러 메시지로 해당 섹션을 찾아갈 수 있도록 인덱스를 두었습니다.

| 증상 / 에러 메시지 | 섹션 |
|------------------|------|
| `NCCL error ... unhandled cuda error` 또는 첫 스텝에서 hang | [§1](#1-nccl-초기화-실패--타임아웃) |
| `Can't pickle local object`, `cannot pickle '_thread.RLock'` | [§2](#2-pickle-에러-torchdistributorrunfn-) |
| `401 Unauthorized`, `Could not find experiment` (child) | [§2-1](#2-1-child-프로세스에서-mlflow가-인증-실패) |
| `CUDA out of memory` at first forward | [§3](#3-oom-at-first-forward) |
| DDP backward에서 일부 rank만 멈춤 / hang | [§4](#4-ddp에서-학습이-멈춘-것처럼-보입니다) |
| MLflow에 같은 metric이 rank 수만큼 중복 | [§5](#5-mlflow가-중복-로그를-만듭니다) |
| UC Volume 저장이 매우 느림 | [§6](#6-uc-volume-저장이-너무-느립니다) |
| Multi-node `local_mode=False` 가 실패 | [§7](#7-serverless-gpu에서-multi-node가-안-됩니다) |
| `@distributed` 안에서 `dbutils` 호출 실패 | [§8](#8-distributed-안에서-dbutils를-못-씁니다) |
| `Skip logging GPU metrics` 만 찍히고 GPU 차트 비어 있음 | [§9](#9-mlflow가-skip-logging-gpu-metrics-만-찍고-gpu-차트가-비어-있습니다) |
| Multi-node throughput 이 기대치보다 낮음 | [§10](#10-multi-node-ddp인데-throughput이-안-올라갑니다) |
| Worker rank의 stderr/NCCL 로그를 어디서 보는지 모름 | [§11](#11-multi-node-worker의-로그를-어디서-보는가) |
| `Py4JError: ... callbackServer`, 두 번째 `TorchDistributor.run` 호출이 죽음 | [§12](#12-같은-노트북에서-torchdistributorrun-연속-호출-시-py4j-callback-단절) |

## 1. NCCL 초기화 실패 / 타임아웃

증상은 `NCCL error ... unhandled cuda error`가 출력되거나 첫 스텝에서 멈추는 형태로 나타납니다.

원인은 다음 중 하나입니다.

- 노드 간 통신 포트가 막혀 있습니다.
- `MASTER_ADDR`/`MASTER_PORT`가 잘못 설정되어 있습니다.
- IB/EFA가 활성화되지 않아 default ethernet으로 fallback됩니다.

해결책은 다음과 같습니다.

- TorchDistributor를 쓰면 위 환경변수가 자동으로 설정되므로 직접 만들지 않습니다.
- 로그의 `NCCL INFO` 라인을 보고 어떤 transport를 쓰는지 확인합니다.
- 필요하면 `NCCL_DEBUG=INFO`, `NCCL_SOCKET_IFNAME=eth0` 등을 조정합니다.

## 2. pickle 에러 (TorchDistributor.run(fn, ...))

증상은 `Can't pickle local object`나 `cannot pickle '_thread.RLock'`입니다.

원인은 `fn`이 closure로 driver 변수(SparkSession, MLflow client 등)를 캡처했기 때문입니다.

해결책은 다음 세 가지를 함께 적용합니다.

- 학습 함수는 **모듈 최상위 또는 별도 `.py` 파일에 정의**합니다.
- 인자로는 primitives(str, int, list)만 넘깁니다.
- SparkSession은 함수 내부에서 `SparkSession.builder.getOrCreate()`로 다시 얻습니다.

## 2-1. child 프로세스에서 MLflow가 인증 실패

증상은 `mlflow.start_run(run_id=...)`이 `401 Unauthorized` 또는 `Could not find experiment`로 떨어지는 것입니다.

원인은 TorchDistributor가 띄운 worker 프로세스가 driver의 Databricks 자격증명을 자동 상속하지 않기 때문입니다.

해결책은 driver에서 host/token을 명시적으로 읽어 학습 함수의 인자로 넘기고, child 안에서 환경변수에 다시 설정하는 것입니다.

```python
# Driver cell
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
db_host = context.extraContext().apply("api_url")
db_token = context.apiToken().get()

TorchDistributor(...).run(train_fn, db_host=db_host, db_token=db_token, ...)
```

```python
def train_fn(db_host, db_token, run_id, ...):
    import os
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token
    # 이제 mlflow가 자격증명을 찾을 수 있습니다
```

## 3. OOM at first forward

증상은 첫 forward에서 곧바로 OOM이 발생하는 것입니다.

원인은 다음 중 하나입니다.

- `per_device_batch_size`가 너무 큽니다.
- 임베딩 테이블이 GPU 메모리에서 큰 비중을 차지하는데 배치 크기를 함께 키웠습니다.
- AdamW 옵티마이저 상태(파라미터의 2배)를 빼고 추산했습니다.

해결책은 다음과 같습니다.

1. `per_device_batch_size`를 절반으로 줄이고 점진적으로 늘립니다.
2. `optimizer.zero_grad(set_to_none=True)`로 그래디언트 메모리를 회수합니다.
3. 임베딩이 너무 크면 `env-databricks-environments.md`에서 더 큰 GPU로 옮깁니다.

## 4. DDP에서 학습이 멈춘 것처럼 보입니다

원인은 두 가지를 의심해 볼 수 있습니다.

- rank 간 데이터 분할이 불균등해서 한쪽이 먼저 끝나면 다른 rank가 AllReduce를 기다립니다.
- 모델에 사용되지 않은 파라미터가 있으면 DDP가 backward에서 hang합니다.

해결책은 다음과 같습니다.

- `DistributedSampler(..., drop_last=True)`로 각 rank의 step 수를 똑같이 맞춥니다.
- `DistributedDataParallel(..., find_unused_parameters=True)`는 정말 필요한 경우에만 켜고, 가능하면 미사용 파라미터를 모델에서 제거합니다.

## 5. MLflow가 중복 로그를 만듭니다

원인은 모든 rank가 `mlflow.start_run`을 호출했기 때문입니다.

해결책은 [`ops-mlflow-tracking.md`](ops-mlflow-tracking.md)의 "rank 0만 로깅" 패턴을 적용하는 것입니다.

## 6. UC Volume 저장이 너무 느립니다

원인은 모든 rank가 매 save_steps마다 동시에 UC Volume에 쓰기 때문입니다.

해결책은 [`data-uc-volumes-checkpoints.md`](data-uc-volumes-checkpoints.md)의 "local disk + copy" 패턴을 적용하고, rank 0만 저장하도록 바꾸는 것입니다.

## 7. Serverless GPU에서 multi-node가 안 됩니다

원인은 Serverless GPU가 single-node 환경이라는 데 있습니다. M×N 셀은 **Classic GPU 클러스터**가 필요합니다.

해결책은 매트릭스의 M×N 셀로 넘어갈 때 Classic 클러스터 사양으로 재배포하는 것입니다.

## 8. `@distributed` 안에서 `dbutils`를 못 씁니다

원인은 child 프로세스에 Databricks driver-side notebook context가 없기 때문입니다.

해결책은 `dbutils` 호출을 모두 driver(노트북) 셀에서 미리 끝내는 것입니다. 학습 함수에는 결과 값(예: 경로 문자열)만 넘깁니다.

## 9. MLflow가 "Skip logging GPU metrics" 만 찍고 GPU 차트가 비어 있습니다

증상: `mlflow.start_run(..., log_system_metrics=True)`로 시작했는데 MLflow UI의 System Metrics 탭에 CPU/메모리/디스크는 보이지만 **GPU utilization/memory/power 차트가 비어 있고**, 로그에 다음 줄만 남습니다.

```
INFO mlflow.system_metrics.system_metrics_monitor: Skip logging GPU metrics. Set logger level to DEBUG for more details.
INFO mlflow.system_metrics.system_metrics_monitor: Started monitoring system metrics.
```

원인: MLflow의 system metrics collector는 GPU 메트릭을 읽을 때 `nvidia-ml-py`(=`pynvml`, NVIDIA Management Library Python 바인딩)를 사용합니다. DBR 17.3 LTS ML에는 `torch`와 CUDA는 들어있지만 `nvidia-ml-py`는 사전 설치되어 있지 않습니다. 모듈 import에 실패하면 MLflow는 silently GPU 수집만 끄고 나머지(CPU·메모리·디스크)는 계속 수집합니다. 학습 자체는 GPU를 정상적으로 사용하고 있고, **차트만** 비어 있는 상태.

해결: `00-setup`의 `%pip install` 셀에 `nvidia-ml-py`를 추가합니다.

```python
%pip install --quiet "lightning==2.5.1" "nvidia-ml-py"
%restart_python
```

multi-node에서의 적용 범위도 짚어 둡니다. `%pip install` + `%restart_python`은 **notebook-scoped 라이브러리**라서 같은 클러스터에 attach된 driver와 모든 worker에 자동 반영됩니다. TorchDistributor(`local_mode=False`)가 띄우는 worker 프로세스도 같은 클러스터 위에서 fresh Python으로 시작하지만, 이 단계에서는 이미 클러스터의 라이브러리 상태에 `nvidia-ml-py`가 포함되어 있으므로 child에서 별도 설치가 필요 없습니다. 단, `cluster.restart`를 거치면 notebook-scoped 라이브러리가 사라지므로 setup 셀은 매 세션마다 재실행해야 합니다.

마지막으로 알아 두면 좋은 점이 두 가지 있습니다.

- MLflow가 인식하는 패키지 이름은 `nvidia-ml-py`(공식, 최신)와 `pynvml`(예전 별칭) 모두입니다. 새로 설치한다면 `nvidia-ml-py`를 권장합니다.
- 같은 차트가 비어 있는 또 다른 원인은 **multi-node에서 driver만 `log_system_metrics=True`** 인 경우입니다(driver는 학습에 참여하지 않음). 해결책은 [`ops-mlflow-tracking.md` §1](ops-mlflow-tracking.md)에 있으며, rank-0 worker가 `mlflow.start_run(run_id=..., log_system_metrics=True)`로 attach하는 패턴입니다.

## 11. multi-node worker의 로그를 어디서 보는가

증상은 다음과 같습니다. `local_mode=False` 학습이 시작은 됐는데 노트북 셀에는 driver의 코디네이션 로그만 보이고, worker rank들의 stderr를 찾지 못해 NCCL 에러나 OOM, 학습 진행 상황을 진단할 수 없습니다.

원리부터 짚어 보면, TorchDistributor가 `local_mode=False`로 띄운 child는 **Spark task**로 실행됩니다([`concepts-torchdistributor-internals.md`](concepts-torchdistributor-internals.md)). 각 task의 stdout/stderr는 해당 worker 노드의 executor log에 쌓입니다.

### 어디서 보는가

로그를 확인할 수 있는 위치는 크게 세 군데입니다.

1. **노트북 셀에 흘러나오는 부분.** TorchDistributor가 child stdout을 driver로 stream하기 때문에 약간의 latency는 있지만 학습 함수 안의 `print(...)`가 결국 노트북 셀에 나타납니다. 모든 rank의 출력이 뒤섞여 나오므로 학습 함수에서 항상 rank prefix를 붙이는 것이 좋습니다.

   ```python
   print(f"[rank={global_rank}] epoch={epoch} loss={loss:.4f}")
   ```

2. **Spark UI의 executor log(가장 정확).** 노트북 우측 상단의 "Spark UI" → Stages 탭에서 진행 중인 **barrier stage**를 클릭하면 각 task의 "stderr"·"stdout" 링크가 나옵니다. rank N의 출력은 task N의 log에 해당하며, task index가 곧 global rank입니다. 학습 함수의 `print`뿐 아니라 NCCL·PyTorch warning, OOM stack trace 등이 모두 여기 있습니다.

3. **Driver log + Cluster log delivery.** Cluster 설정의 "Log delivery"를 활성화하면 driver와 worker 로그가 UC Volume 또는 S3에 영구 저장됩니다. 학습 종료 후 사후 분석에 유용하며, 실시간 디버깅에는 Spark UI가 더 빠릅니다.

### NCCL 로그 켜기

multi-node에서 inter-node bandwidth가 의심되거나 rendezvous가 멈출 때는 NCCL 디버그 로그를 켜는 것이 가장 빠른 진단입니다.

```python
def train_fn(...):
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"   # 선택 — INIT만 보고 싶으면 "INIT"
    dist.init_process_group("nccl")
    ...
```

NCCL이 각 rank의 stderr에 사용 중인 transport(IB / EFA / SOCKET), NIC, ring topology를 출력합니다. **Spark UI executor log → stderr 탭**에서 확인합니다.

### 함정

multi-node 로그 확인에서 자주 빠지는 함정은 다음과 같습니다.

- `local_mode=True`의 child stdout/stderr는 driver의 노트북 셀에 자연스럽게 나오므로 Spark UI를 볼 필요가 없습니다.
- worker가 죽었을 때 driver에는 `Py4JError` 또는 `BarrierTaskException`만 보고되는 경우가 많습니다. 실제 원인은 **worker stderr(Spark UI)**에 있으므로 노트북 셀의 에러만 보고 판단하면 안 됩니다.
- multi-node 학습이 영원히 hang하면 대개 NCCL init 실패입니다. `NCCL_DEBUG=INFO`로 어떤 transport에서 멈췄는지 확인합니다.
- Spark UI는 cluster terminate 후에도 일정 시간 접근할 수 있지만 사라질 수 있습니다. 중요한 로그는 cluster log delivery로 영구화합니다.

## 10. multi-node DDP인데 throughput이 안 올라갑니다

원인은 다음 중 하나일 가능성이 높습니다.

- 모델이 너무 작아 AllReduce 통신 비용이 compute 비중과 비슷해진 상태입니다.
- inter-node bandwidth(EFA 미활성)가 병목입니다.
- batch size가 작아 GPU가 idle 상태입니다.

해결책은 다음 순서로 시도합니다.

- `per_device_batch_size`를 GPU 메모리가 허용하는 최대까지 키웁니다.
- `pin_memory=True`, `num_workers=4` 등 DataLoader 옵션을 점검합니다.
- 그래도 안 오르면 단일 노드로 충분한 워크로드일 수 있음을 인정합니다. 본 쿡북의 M×N 셀은 **패턴 시연**이 1차 목적이라는 점도 함께 고려합니다.

## 12. 같은 노트북에서 `TorchDistributor.run` 연속 호출 시 py4j callback 단절

[`02-script-based/README.md`](../02-script-based/README.md)와 [`03-custom-package-script-based/README.md`](../03-custom-package-script-based/README.md)에서 언급된 현상입니다(DBR 17.3 LTS ML, g5.12xlarge에서 관찰). 근본 원인은 확정되지 않았지만 가능한 가설은 다음과 같습니다.

1. **TorchDistributor가 cloudpickle 직렬화 시 driver-side `SparkContext`/`SparkSession` 참조를 캡처할 수 있습니다.** child가 그 참조를 deserialize하려고 py4j gateway에 접근하면, 두 번째 호출 시점에 gateway가 stale 상태일 수 있습니다.
2. **barrier execution mode가 SparkContext의 `BarrierTaskContext` 상태를 변경**합니다. 연속 호출 시 이전 barrier job의 정리가 늦어지면서 callback channel이 일시적으로 끊길 수 있습니다.
3. **child 프로세스가 종료 시 driver-side py4j callback 서버에 등록된 listener를 정리하지 못하는 경우**가 있어, 누적된 stale callback이 다음 호출의 새 callback과 충돌할 수 있습니다.

증상은 두 번째 `TorchDistributor.run` 호출이 시작은 되지만 학습 중간에 `Py4JError: An error occurred while calling ... callbackServer` 같은 에러로 죽거나, 학습은 끝났는데 driver에서 결과 수집이 되지 않는 형태로 나타납니다.

본 쿡북이 채택한 회피책은 **launcher × 토폴로지별로 노트북을 분리**해 매번 fresh Python interpreter로 시작하는 것입니다. 같은 노트북에서 `dbutils.notebook.exit` 후 재실행해도 되지만 노트북 분리가 더 깔끔합니다.

다른 회피책도 참고삼아 적어 둡니다.

- `spark.conf.set("spark.databricks.python.barrier.enabled", "true")` 명시 (이미 default)
- 두 호출 사이에 `SparkContext._gateway.callback_server.shutdown()` 후 재시작 — fragile해서 권장하지 않음
