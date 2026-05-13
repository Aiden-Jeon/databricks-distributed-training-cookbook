# MLflow Tracking

분산 학습에서 MLflow를 사용하는 패턴을 정리합니다. 본 쿡북은 **MLflow 3.0+** 기능(system metrics, per-epoch `LoggedModel`, dataset linking)을 명시적으로 사용합니다. 학습 후 Unity Catalog Model Registry에 등록하는 흐름은 [`ops-uc-model-registry.md`](ops-uc-model-registry.md)에 분리되어 있습니다.

## 한눈에 보기

| 패턴 | rank 0 가드 | `log_system_metrics` 위치 | 비고 |
|------|-------------|---------------------------|------|
| 단일 GPU (PyTorch 루프 직접) | 단일 프로세스라 불필요 | driver의 `start_run` | 가장 단순 |
| TorchDistributor `local_mode=True` (1×1 / 1×N) | 명시 필요 (`RANK == 0`) | driver의 `start_run` (with-block이 `.run()`을 감싸 둠) | driver=worker 동일 머신 |
| TorchDistributor `local_mode=False` (M×N) | 명시 필요 (`RANK == 0`) | rank-0 **worker**의 `start_run(run_id=...)` | driver는 학습 미참여 |
| Lightning 직접 실행 (1×1 / 1×N) | 자동 (`MLFlowLogger`) | driver의 `start_run` (logger와 별도 호출) | logger에 옵션 없음 |
| Lightning + TorchDistributor (M×N) | 자동 (`MLFlowLogger`) | rank-0 **worker**의 `start_run(run_id=...)` | logger + 별도 attach |
| Accelerate (`accelerate launch`) | 명시 필요 (`RANK == 0`) | rank-0 **worker**의 `start_run(run_id=...)` | subprocess, env vars 명시 전달 |

## 기본 원칙

분산 학습에서 MLflow를 안전하게 쓰려면 다음 세 가지를 지키면 됩니다.

- **rank 0만 로깅합니다.** 모든 rank가 동시에 `mlflow.log_metric`을 호출하면 race가 일어납니다.
- 실험(`experiment`)과 run은 driver(=rank 0)에서 시작하고 child 프로세스에 `run_id`를 전달합니다.
- Lightning `Trainer`의 `MLFlowLogger`는 자동으로 rank 0에서만 기록되므로 별도 가드가 필요 없습니다. 직접 PyTorch 루프를 쓰는 경우에는 가드를 명시합니다.

## MLflow 3.0+ 사용 기능

### 시스템 메트릭 (`log_system_metrics`)

`mlflow.start_run(..., log_system_metrics=True)`로 시스템 메트릭(CPU/GPU/메모리/네트워크/디스크)이 자동 수집됩니다. MLflow UI의 System Metrics 탭에서 GPU 활용도, 데이터 로더 병목, 네트워크 사용량을 볼 수 있습니다.

Multi-node TorchDistributor(`local_mode=False`)에서는 driver-side만 켜면 idle driver 메트릭만 잡힙니다. 학습 노드 메트릭을 보려면 **rank-0 worker가 `start_run(run_id=..., log_system_metrics=True)`로 attach할 때 함께 켜야 합니다**(아래 패턴 3, 5, 6 참조). Single-node(`local_mode=True`)는 driver와 worker가 같은 머신이라 host 메트릭이 동일하므로 driver-side `start_run(..., log_system_metrics=True)`의 with-block으로 `.run()`을 감싸 두면 충분합니다(패턴 2). Lightning은 `MLFlowLogger`에 `log_system_metrics` 옵션이 없으므로 `MLFlowLogger(run_id=...)`와 **별도로** `mlflow.start_run(run_id=..., log_system_metrics=True)`를 호출해 메트릭 스레드를 띄워야 합니다(패턴 4, 5).

### 에폭별 체크포인트 (`LoggedModel`)

기존 MLflow에서는 `log_model`을 여러 번 호출하면 artifact가 중복됐습니다. MLflow 3.0+는 **하나의 run 안에 여러 LoggedModel**을 가질 수 있어, 각 epoch의 checkpoint를 모두 보존하고 학습 종료 후 가장 좋은 것을 선택할 수 있습니다.

```python
import mlflow

# 시그니처 생성 (UC 등록을 위해 필수)
sample_input = next(iter(train_loader))[0][:5].cpu().numpy()
model.eval()
with torch.no_grad():
    sample_output = model(*[torch.tensor(x).to(device) for x in sample_input]).cpu().numpy()
signature = mlflow.models.infer_signature(sample_input, sample_output)

for epoch in range(NUM_EPOCHS):
    # ... 학습 / 검증 ...
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        name=f"recommender-epoch-{epoch+1}",
        params={"epoch": epoch + 1, "architecture": "TwoTowerMLP"},
        step=epoch,
        signature=signature,
        input_example=sample_input,
    )

    # 메트릭을 이 LoggedModel과 dataset에 연결
    mlflow.log_metric(
        "val/auc", val_auc,
        step=epoch,
        model_id=model_info.model_id,
        dataset=val_mlflow_dataset,                  # 아래 "데이터셋 연결" 참고
    )
```

학습 종료 후 best LoggedModel을 골라 UC에 등록하는 흐름은 [`ops-uc-model-registry.md`](ops-uc-model-registry.md)를 참조하세요.

### 데이터셋 연결 (`mlflow.data`)

학습 데이터를 MLflow dataset 객체로 등록하면, 메트릭이 어느 데이터셋에서 측정됐는지 추적할 수 있습니다.

```python
import mlflow

train_dataset = mlflow.data.from_pandas(train_pd, name="train")
val_dataset = mlflow.data.from_pandas(val_pd, name="validation")

mlflow.log_metric("train/loss", loss, step=epoch, dataset=train_dataset)
```

## 패턴별 코드

표의 6개 패턴을 1:1로 풀어 둡니다. 모든 패턴은 동일한 헬퍼 가정(`set_experiment(...)`, `db_host`/`db_token`은 `dbutils` 컨텍스트에서 추출)으로 시작합니다.

```python
# 모든 패턴에서 공통
import mlflow
mlflow.set_experiment("/Users/<email>/recommender")

# 분산 패턴(2~6)에서는 child에 전달할 자격증명을 driver에서 추출
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
db_host = context.extraContext().apply("api_url")
db_token = context.apiToken().get()
```

### 1. 단일 GPU (PyTorch 루프 직접)

launcher 없이 driver에서 단일 프로세스로 학습. rank 가드 불필요, 시스템 메트릭은 driver 한 군데에서만 켜면 충분합니다.

```python
with mlflow.start_run(run_name="recommender-1x1", log_system_metrics=True) as run:
    run_id = run.info.run_id

    mlflow.log_params({
        "n_users": N_USERS, "n_items": N_ITEMS, "emb_dim": EMB_DIM,
        "batch_size": BATCH_SIZE, "lr": LR,
    })

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss, val_auc = evaluate(model, val_loader, device)

        mlflow.log_metric("train/loss", train_loss, step=epoch)
        mlflow.log_metric("val/loss", val_loss, step=epoch)
        mlflow.log_metric("val/auc", val_auc, step=epoch)
```

`mlflow.pytorch.autolog()`를 호출하면 옵티마이저 step, learning rate, 모델 그래프 등을 자동으로 기록합니다. 분산 패턴에서는 반드시 **rank 0에서만** 호출해야 합니다.

### 2. TorchDistributor `local_mode=True` (1×1 / 1×N)

Single-node. driver 머신 안에서 `num_processes`개 child가 학습하므로 host 메트릭은 driver-side에서 잡아도 동일합니다. **driver의 with-block이 `.run()`을 감싸도록 두면** system metrics 스레드가 학습 기간 내내 살아 있습니다. child는 fresh interpreter이므로 `DATABRICKS_HOST/TOKEN`은 명시적으로 전달합니다([`env-auth.md`](env-auth.md)).

```python
# train_fn (모듈 최상위 또는 별도 .py)
def train_fn(run_id, db_host, db_token, ...):
    import os, mlflow
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token

    rank = int(os.environ["RANK"])
    if rank == 0:
        mlflow.start_run(run_id=run_id)        # driver-side system metrics 스레드가 host 메트릭을 잡고 있음
    # ... DDP 학습 ...
    if rank == 0:
        mlflow.end_run()
```

```python
# Driver (notebook cell)
from pyspark.ml.torch.distributor import TorchDistributor

with mlflow.start_run(run_name="recommender-1xN", log_system_metrics=True) as run:
    run_id = run.info.run_id
    TorchDistributor(num_processes=4, local_mode=True, use_gpu=True).run(
        train_fn, run_id=run_id, db_host=db_host, db_token=db_token,
        # ... 기타 학습 인자 ...
    )
```

### 3. TorchDistributor `local_mode=False` (M×N)

Multi-node. driver는 학습에 참여하지 않으므로 system metrics는 **rank-0 worker가 attach할 때 함께 켜야** 학습 노드 GPU 사용률이 잡힙니다. driver의 with-block은 `.run()` **전에 빠져나와도** 됩니다(`start_run(run_id=...)`은 종료된 run에도 append 가능; 아래 "Databricks 자동 처리" 참조).

```python
# train_fn (모듈 최상위 또는 별도 .py)
def train_fn(run_id, db_host, db_token, ...):
    import os, mlflow
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token

    rank = int(os.environ["RANK"])
    if rank == 0:
        # multi-node는 driver가 학습 미참여 → worker 측 attach 시점에 log_system_metrics를 켬
        mlflow.start_run(run_id=run_id, log_system_metrics=True)
    # ... DDP 학습 ...
    if rank == 0:
        mlflow.end_run()
```

```python
# Driver (notebook cell)
from pyspark.ml.torch.distributor import TorchDistributor

with mlflow.start_run(run_name="recommender-mxn") as run:
    run_id = run.info.run_id

TorchDistributor(num_processes=8, local_mode=False, use_gpu=True).run(
    train_fn, run_id=run_id, db_host=db_host, db_token=db_token,
    # ... 기타 학습 인자 ...
)
```

### 4. Lightning 직접 실행 (1×1 / 1×N)

Single-node. driver에서 `Trainer.fit()`을 직접 호출. `MLFlowLogger`가 rank-0 가드를 자동 처리합니다. `MLFlowLogger`에는 `log_system_metrics` 옵션이 없으므로 driver에서 **별도로** `mlflow.start_run(..., log_system_metrics=True)`를 띄우고 logger를 `run_id`로 같은 run에 attach합니다.

```python
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger

with mlflow.start_run(run_name="recommender-lightning-1xN", log_system_metrics=True) as run:
    run_id = run.info.run_id

    logger = MLFlowLogger(
        run_id=run_id,
        tracking_uri="databricks",
        log_model=True,                          # epoch별 checkpoint 자동 등록
    )
    trainer = pl.Trainer(devices=4, num_nodes=1, strategy="ddp", logger=logger, ...)
    trainer.fit(model, train_loader, val_loader)
```

### 5. Lightning + TorchDistributor (M×N)

Multi-node. Lightning Trainer를 train_fn 안에서 띄우고 TorchDistributor가 worker에 분산합니다. rank-0 worker가 별도 `start_run(run_id=..., log_system_metrics=True)`로 attach하고, `MLFlowLogger(run_id=...)`로 logger도 같은 run에 연결합니다.

```python
def train_fn(run_id, db_host, db_token, ...):
    import os, mlflow
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import MLFlowLogger

    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token

    rank = int(os.environ["RANK"])
    if rank == 0:
        # logger와 별개로 system metrics 스레드를 띄움
        mlflow.start_run(run_id=run_id, log_system_metrics=True)

    logger = MLFlowLogger(run_id=run_id, tracking_uri="databricks", log_model=True)
    trainer = pl.Trainer(devices=N, num_nodes=M, strategy="ddp", logger=logger, ...)
    trainer.fit(model, train_loader, val_loader)

    if rank == 0:
        mlflow.end_run()
```

```python
# Driver (notebook cell)
from pyspark.ml.torch.distributor import TorchDistributor

with mlflow.start_run(run_name="recommender-lightning-mxn") as run:
    run_id = run.info.run_id

TorchDistributor(num_processes=M*N, local_mode=False, use_gpu=True).run(
    train_fn, run_id=run_id, db_host=db_host, db_token=db_token,
    # ... 기타 학습 인자 ...
)
```

### 6. Accelerate (`accelerate launch`)

`accelerate launch script.py` 형태. subprocess라 driver Python 메모리는 상속되지 않고 env vars만 OS 레벨로 전달됩니다. TD M×N과 동일한 attach 패턴을 적용합니다 — driver에서 run을 만들고 `run_id`/자격증명을 env var로 넘긴 뒤, script 안에서 `RANK == 0`일 때 `start_run(run_id=..., log_system_metrics=True)`로 attach합니다.

```python
# Driver (notebook cell)
import os, subprocess

with mlflow.start_run(run_name="recommender-accelerate") as run:
    run_id = run.info.run_id

env = os.environ.copy()
env.update({
    "MLFLOW_RUN_ID": run_id,
    "DATABRICKS_HOST": db_host,
    "DATABRICKS_TOKEN": db_token,
})
subprocess.run(["accelerate", "launch", "train.py"], env=env, check=True)
```

```python
# train.py (script)
import os, mlflow

rank = int(os.environ["RANK"])
if rank == 0:
    mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"], log_system_metrics=True)
# ... Accelerator() + 학습 루프 ...
if rank == 0:
    mlflow.end_run()
```

## Databricks 자동 처리

위 패턴은 일반 MLflow와 동일하지만, Databricks 위에서는 두 가지가 자동·특수 처리됩니다.

### `tracking_uri="databricks"` 자동 라우팅

Databricks 노트북이나 Job에서 시작된 Python은 환경변수 `MLFLOW_TRACKING_URI=databricks`가 미리 세팅되어 있어, 별도 설정 없이도 워크스페이스 MLflow로 라우팅됩니다. `MLFlowLogger(tracking_uri="databricks")`의 명시는 redundant하지만 child가 환경변수를 못 받는 시나리오 대비로 권장합니다. TorchDistributor child도 환경변수는 상속하므로 `MLFLOW_TRACKING_URI`는 자동 전달됩니다 — child가 인증에 실패한다면 원인은 URI가 아니라 토큰입니다([`env-auth.md`](env-auth.md)).

### with-block을 빠져나와도 attach 가능

```python
with mlflow.start_run(...) as run:
    run_id = run.info.run_id
# 빠져나옴 → driver-side run은 FINISHED. 그런데 child에서 attach가 가능
TorchDistributor(...).run(train_fn, run_id=run_id)
```

MLflow의 `start_run(run_id=...)`은 **이미 종료된 run에도 metric을 추가로 append할 수 있게** 설계됐습니다. UI는 latest metric까지 그대로 보여 줍니다. driver의 `end_run`은 "현재 thread의 active run을 닫는다"는 의미일 뿐, run 객체 자체는 서버에 남아 있습니다.

## Experiment 경로 — 사용자 vs SP

실행 주체에 따라 experiment 경로를 어디에 두어야 하는지 정리하면 다음과 같습니다.

| 실행 주체 | 권장 EXPERIMENT_PATH |
|----------|--------------------|
| Interactive 노트북 (사용자) | `/Users/{username}/recommender-...` |
| Job, `run_as = 사용자` | 동일 |
| Job, `run_as = service principal` | `/Shared/recommender-...` (SP는 personal 폴더 없음) |

본 쿡북의 `EXPERIMENT_PATH = f"/Users/{USERNAME}/..."`는 사용자 기준입니다. SP로 돌릴 때는 setup 노트북의 경로를 교체합니다. 자세한 내용은 [`env-auth.md`](env-auth.md)를 참조하세요.

## 참고

- [HF Transformers on Databricks](https://docs.databricks.com/aws/en/machine-learning/train-model/huggingface/)
- [MLflow autolog](https://mlflow.org/docs/latest/tracking/autolog.html)
- [MLflow 3.0 LoggedModel](https://mlflow.org/docs/latest/model)
- [MLflow system metrics](https://mlflow.org/docs/latest/system-metrics)
