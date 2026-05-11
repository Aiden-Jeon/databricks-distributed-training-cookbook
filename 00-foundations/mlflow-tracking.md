# MLflow Tracking

분산 학습에서 MLflow를 사용하는 패턴. 본 쿡북은 **MLflow 3.0+** 기능(system metrics, per-epoch `LoggedModel`, dataset linking)을 명시적으로 사용한다.

## 기본 원칙

- **rank 0만 로깅한다.** 모든 rank가 동시에 `mlflow.log_metric`을 호출하면 race가 일어난다.
- 실험(`experiment`)과 run은 driver(=rank 0)에서 시작하고 child 프로세스에 `run_id`를 전달한다.
- Lightning `Trainer`의 `MLFlowLogger`는 자동으로 rank 0에서만 기록되므로 별도 가드가 필요 없다. 직접 PyTorch 루프를 쓰는 경우 가드를 명시한다.

## PyTorch 루프 + MLflow 직접 호출 (단일 GPU)

가장 단순한 형태. 시스템 메트릭(CPU/GPU/메모리/네트워크/디스크)을 자동 수집하고 모든 epoch을 하나의 run에 묶는다.

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

`mlflow.pytorch.autolog()`를 호출하면 옵티마이저 step·learning rate·모델 그래프 등을 자동으로 기록한다. **rank 0에서만** 호출한다.

## MLflow 3.0+ 핵심 기능

### 1. `log_system_metrics=True`

`mlflow.start_run(..., log_system_metrics=True)`로 시스템 메트릭이 자동 수집된다. MLflow UI → System Metrics 탭에서 GPU 활용도, 메모리, 네트워크, 디스크 I/O를 볼 수 있다. 분산 학습에서 GPU가 idle 상태인지(데이터 로더 병목) 진단할 때 핵심.

### 2. 에폭별 `LoggedModel` (`mlflow.pytorch.log_model(..., step=...)`)

기존 MLflow에서는 `log_model`을 여러 번 호출하면 artifact가 중복됐다. MLflow 3.0+는 **하나의 run 안에 여러 LoggedModel**을 가질 수 있다. 각 epoch의 checkpoint를 모두 보존하고, 학습 종료 후 가장 좋은 것을 선택할 수 있다.

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

학습 데이터를 MLflow dataset 객체로 등록하면, 메트릭이 어느 데이터셋에서 측정됐는지 추적된다.

```python
import mlflow

train_dataset = mlflow.data.from_pandas(train_pd, name="train")
val_dataset = mlflow.data.from_pandas(val_pd, name="validation")

# log_metric 호출 시 dataset 인자로 연결
mlflow.log_metric("train/loss", loss, step=epoch, dataset=train_dataset)
```

## TorchDistributor / `@distributed` 사용 시

함수 안에서 새로 `mlflow.start_run`을 하지 말고, **driver에서 run_id를 넘긴 뒤 child에서 `mlflow.start_run(run_id=run_id)`로 attach**한다.

또한 child 프로세스는 driver의 Databricks 자격증명을 자동 상속하지 않으므로, `DATABRICKS_HOST`/`DATABRICKS_TOKEN`을 명시적으로 전달한다.

```python
# Driver (notebook cell)
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
db_host = context.extraContext().apply("api_url")
db_token = context.apiToken().get()

with mlflow.start_run(run_name="recommender-mxn", log_system_metrics=True) as run:
    run_id = run.info.run_id
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
        mlflow.start_run(run_id=run_id)
    # ... DDP 학습 ...
    if rank == 0:
        mlflow.end_run()
```

## Lightning + MLflow

`MLFlowLogger`를 `Trainer(logger=...)`로 명시 주입한다. rank-0 가드는 Lightning이 알아서 처리한다.

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

학습 후 best checkpoint를 UC Model Registry에 등록. 시그니처가 있어야 한다 (위 #2 참고).

```python
uc_model_name = "main.recsys.two_tower_mlp"
model_uri = f"runs:/{run_id}/model"          # 또는 best LoggedModel의 model_uri

registered = mlflow.register_model(model_uri, uc_model_name)
print(f"Registered {uc_model_name} v{registered.version}")
```

이 단계는 각 셀의 `eval_and_register` 노트북에서 처리.

## 참고

- HF Transformers on Databricks: https://docs.databricks.com/aws/en/machine-learning/train-model/huggingface/
- MLflow autolog: https://mlflow.org/docs/latest/tracking/autolog.html
- MLflow 3.0 LoggedModel: https://mlflow.org/docs/latest/model
- MLflow system metrics: https://mlflow.org/docs/latest/system-metrics
- 패턴 출처: [`99-references/snippets/fashion_recommendations/train_simple_mlp.ipynb`](../99-references/snippets/fashion_recommendations/train_simple_mlp.ipynb)
