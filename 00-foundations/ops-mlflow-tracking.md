# MLflow Tracking

분산 학습에서 MLflow를 사용하는 패턴. 본 쿡북은 **MLflow 3.0+** 기능(system metrics, per-epoch `LoggedModel`, dataset linking)을 명시적으로 사용합니다.

## 기본 원칙

- **rank 0만 로깅합니다.** 모든 rank가 동시에 `mlflow.log_metric`을 호출하면 race가 일어납니다.
- 실험(`experiment`)과 run은 driver(=rank 0)에서 시작하고 child 프로세스에 `run_id`를 전달합니다.
- Lightning `Trainer`의 `MLFlowLogger`는 자동으로 rank 0에서만 기록되므로 별도 가드가 필요 없습니다. 직접 PyTorch 루프를 쓰는 경우 가드를 명시합니다.

## PyTorch 루프 + MLflow 직접 호출 (단일 GPU)

가장 단순한 형태. 시스템 메트릭(CPU/GPU/메모리/네트워크/디스크)을 자동 수집하고 모든 epoch을 하나의 run에 묶습니다.

```python
import os
import mlflow

mlflow.set_experiment("/Users/<email>/recommender-1x1")

with mlflow.start_run(
    run_name="recommender-1x1",
    log_system_metrics=True,                # MLflow 3.0+: 시스템 메트릭 자동 수집
) as run:
    run_id = run.info.run_id

    # 하이퍼파라미터
    mlflow.log_params({
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "emb_dim": EMB_DIM,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    })

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss, val_auc = evaluate(model, val_loader, device)

        # 에폭별 메트릭
        mlflow.log_metric("train/loss", train_loss, step=epoch)
        mlflow.log_metric("val/loss", val_loss, step=epoch)
        mlflow.log_metric("val/auc", val_auc, step=epoch)
```

`mlflow.pytorch.autolog()`를 호출하면 옵티마이저 step·learning rate·모델 그래프 등을 자동으로 기록합니다. **rank 0에서만** 호출합니다.

## MLflow 3.0+ 핵심 기능

### 1. `log_system_metrics=True`

`mlflow.start_run(..., log_system_metrics=True)`로 시스템 메트릭이 자동 수집됩니다. MLflow UI → System Metrics 탭에서 GPU 활용도, 메모리, 네트워크, 디스크 I/O를 볼 수 있습니다. 분산 학습에서 GPU가 idle 상태인지(데이터 로더 병목) 진단할 때 핵심.

**Multi-node TorchDistributor 함정**: `local_mode=False`에서는 학습이 worker 노드에서만 실행되고 driver는 코디네이션만 담당합니다. 따라서 driver-side `log_system_metrics=True`만으론 idle driver의 메트릭만 잡히고 실제 GPU 사용률은 비어 있습니다. 실제 학습 노드의 메트릭을 보려면 **rank-0 worker가 `mlflow.start_run(run_id=..., log_system_metrics=True)`로 attach할 때 함께 켜야** 합니다. single-node(`local_mode=True`)는 driver와 worker가 같은 머신이라 driver-side만으로 충분합니다. Lightning을 쓰는 경우 `MLFlowLogger`는 `log_system_metrics` 옵션이 없으므로, worker 함수 안에서 `MLFlowLogger(run_id=...)`와 별도로 `mlflow.start_run(run_id=..., log_system_metrics=True)`를 호출해 메트릭 스레드를 띄워야 합니다 (아래 TorchDistributor 섹션 예제 참고).

### 2. 에폭별 `LoggedModel` (`mlflow.pytorch.log_model(..., step=...)`)

기존 MLflow에서는 `log_model`을 여러 번 호출하면 artifact가 중복됐습니다. MLflow 3.0+는 **하나의 run 안에 여러 LoggedModel**을 가질 수 있습니다. 각 epoch의 checkpoint를 모두 보존하고, 학습 종료 후 가장 좋은 것을 선택할 수 있습니다.

```python
import mlflow

# 시그니처 생성 (Unity Catalog 등록을 위해 필수)
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
        dataset=val_mlflow_dataset,                  # 아래 #3 참고
    )
```

학습 종료 후 best 모델을 찾으려면:

```python
from mlflow import MlflowClient
client = MlflowClient()
run = client.get_run(run_id)
model_outputs = run.outputs.model_outputs  # 모든 LoggedModel
# ... 메트릭과 묶어 정렬 ...
```

### 3. Dataset linking (`mlflow.data.from_pandas`)

학습 데이터를 MLflow dataset 객체로 등록하면, 메트릭이 어느 데이터셋에서 측정됐는지 추적됩니다.

```python
import mlflow

train_dataset = mlflow.data.from_pandas(train_pd, name="train")
val_dataset = mlflow.data.from_pandas(val_pd, name="validation")

# log_metric 호출 시 dataset 인자로 연결
mlflow.log_metric("train/loss", loss, step=epoch, dataset=train_dataset)
```

## TorchDistributor / `@distributed` 사용 시

함수 안에서 새로 `mlflow.start_run`을 하지 말고, **driver에서 run_id를 넘긴 뒤 child에서 `mlflow.start_run(run_id=run_id)`로 attach**합니다.

또한 child 프로세스는 driver의 Databricks 자격증명을 자동 상속하지 않으므로, `DATABRICKS_HOST`/`DATABRICKS_TOKEN`을 명시적으로 전달합니다.

```python
# Driver (notebook cell)
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
db_host = context.extraContext().apply("api_url")
db_token = context.apiToken().get()

with mlflow.start_run(run_name="recommender-mxn", log_system_metrics=True) as run:
    run_id = run.info.run_id

# with-block을 빠져나가면 driver-side run은 FINISHED 상태가 되지만, run_id로 child에서 metric을 계속 append할 수 있습니다.
TorchDistributor(num_processes=8, local_mode=False, use_gpu=True).run(
    train_fn,
    run_id=run_id,
    db_host=db_host,
    db_token=db_token,
    # ... 기타 학습 인자 ...
)
```

```python
# train_fn (모듈 최상위 또는 별도 .py)
def train_fn(run_id, db_host, db_token, ...):
    import os, mlflow
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token

    rank = int(os.environ["RANK"])
    if rank == 0:
        # multi-node(local_mode=False)에서는 driver가 학습에 참여하지 않으므로
        # 실제 GPU 메트릭을 잡으려면 worker 측 attach 시점에 log_system_metrics를 켭니다.
        mlflow.start_run(run_id=run_id, log_system_metrics=True)
    # ... DDP 학습 ...
    if rank == 0:
        mlflow.end_run()
```

## Lightning + MLflow

`MLFlowLogger`를 `Trainer(logger=...)`로 명시 주입합니다. rank-0 가드는 Lightning이 알아서 처리합니다.

```python
from lightning.pytorch.loggers import MLFlowLogger

logger = MLFlowLogger(
    experiment_name="/Users/<email>/recommender-lightning",
    run_name="recommender-1xn",
    tracking_uri="databricks",
    log_model=True,                          # epoch별 checkpoint 자동 등록
)
trainer = pl.Trainer(logger=logger, ...)
```

## Unity Catalog 모델 등록

학습 후 best checkpoint를 UC Model Registry에 등록. 시그니처가 있어야 합니다 (위 #2 참고).

```python
uc_model_name = "main.distributed_cookbook.two_tower_mlp"
model_uri = f"runs:/{run_id}/model"          # 또는 best LoggedModel의 model_uri

registered = mlflow.register_model(model_uri, uc_model_name)
print(f"Registered {uc_model_name} v{registered.version}")
```

이 단계는 각 셀의 `eval_and_register` 노트북에서 처리.

## Databricks 환경에서의 특수성

위 패턴은 모두 일반 MLflow와 동일하지만, Databricks 위에서는 몇 가지 부분이 자동/특수 처리됩니다. 분산 학습에서 자주 헷갈리는 지점만 정리.

### `tracking_uri="databricks"` 가 자동인 이유

Databricks 노트북·Job에서 시작된 Python은 환경변수 `MLFLOW_TRACKING_URI=databricks` 가 미리 세팅되어 있습니다. 그래서:

- `mlflow.start_run(...)` 만 호출해도 자동으로 워크스페이스 MLflow에 기록
- `MLFlowLogger(tracking_uri="databricks")` 의 명시도 redundant하지만 child 프로세스 안전성 차원에서 권장 (child가 환경변수를 못 받는 시나리오 대비)

TorchDistributor child는 fresh interpreter이지만 환경변수는 상속하므로 `MLFLOW_TRACKING_URI` 도 자동 전달됩니다. 그래도 child가 인증 자체에 실패하는 이유는 토큰이지 URI가 아닙니다 ([`env-auth.md`](env-auth.md)).

### Driver의 with-block 이 빠지면 어떻게 attach가 가능한가

`ops-mlflow-tracking.md` 의 multi-node 패턴은 직관에 반합니다:

```python
with mlflow.start_run(...) as run:
    run_id = run.info.run_id
# with-block을 빠져나옴 → driver-side run은 FINISHED 상태

# 그런데 child에서 attach가 가능
TorchDistributor(...).run(train_fn, run_id=run_id)
```

원리: MLflow의 `start_run(run_id=...)` 은 **이미 종료된 run에도 metric을 추가로 append할 수 있게** 설계됐습니다 (MLflow 1.x 이래의 동작). UI는 latest metric까지 그대로 보여줍니다. driver의 `end_run` 은 단지 "현재 thread의 active run을 닫는다" 일 뿐, run 객체 자체는 서버에 남아 있습니다.

→ MLflow 3.0+ 에서도 같은 동작. system_metrics만 child에서 별도 켜야 한다는 점이 다를 뿐 ([`ops-mlflow-tracking.md` §1](ops-mlflow-tracking.md)의 multi-node 함정).

### Unity Catalog Model Registry 권한

학습 종료 후 `mlflow.register_model(uri, "main.distributed_cookbook.two_tower_mlp")` 가 자주 막히는 이유는 권한입니다. 필요한 grant:

```sql
GRANT USE CATALOG ON CATALOG main TO `<user-or-sp>`;
GRANT USE SCHEMA ON SCHEMA main.distributed_cookbook TO `<user-or-sp>`;
GRANT CREATE MODEL ON SCHEMA main.distributed_cookbook TO `<user-or-sp>`;
```

흔한 에러: `PERMISSION_DENIED: User does not have CREATE MODEL on schema`. 자세한 권한 모델은 [`env-auth.md`](env-auth.md) §"권한 함정".

`mlflow.set_registry_uri("databricks-uc")` 가 UC Model Registry로 라우팅하는 핵심 호출 (DBR 17.3 LTS ML은 default가 UC). 워크스페이스 registry로 등록하려면 `"databricks"` 로 명시 — 본 쿡북은 UC 등록을 가정.

### Experiment 경로 — 사용자 vs SP

| 실행 주체 | 권장 EXPERIMENT_PATH |
|----------|--------------------|
| Interactive 노트북 (사용자) | `/Users/{username}/recommender-...` |
| Job, `run_as = 사용자` | 동일 |
| Job, `run_as = service principal` | `/Shared/recommender-...` (SP는 personal 폴더 없음) |

본 쿡북의 `EXPERIMENT_PATH = f"/Users/{USERNAME}/..."` 는 사용자 기준. SP로 돌릴 때는 setup 노트북의 경로를 교체 — 자세한 내용은 [`env-auth.md`](env-auth.md).

## 참고

- HF Transformers on Databricks: https://docs.databricks.com/aws/en/machine-learning/train-model/huggingface/
- MLflow autolog: https://mlflow.org/docs/latest/tracking/autolog.html
- MLflow 3.0 LoggedModel: https://mlflow.org/docs/latest/model
- MLflow system metrics: https://mlflow.org/docs/latest/system-metrics
- Unity Catalog Model Registry: https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/index
