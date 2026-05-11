# MLflow Tracking

분산 학습에서 MLflow를 사용하는 패턴.

## 기본 원칙

- **rank 0만 로깅한다.** 모든 rank가 동시에 `mlflow.log_metric`을 호출하면 race 가 일어난다.
- 실험(`experiment`)과 run은 driver(=rank 0)에서 시작하고 child 프로세스에 run_id를 전달한다.
- Lightning `Trainer`의 `MLFlowLogger`는 자동으로 rank 0에서만 기록되므로 별도 가드가 필요 없다. 직접 PyTorch 루프를 쓰는 경우 가드를 명시한다.

## PyTorch 루프 + MLflow 직접 호출

```python
import os
import mlflow

if int(os.environ.get("RANK", "0")) == 0:
    mlflow.set_experiment("/Users/<email>/distributed-training")
    mlflow.start_run(run_name="recommender-1xn-ddp")

for step, batch in enumerate(loader):
    loss = train_step(batch)
    if int(os.environ.get("RANK", "0")) == 0 and step % 50 == 0:
        mlflow.log_metric("train/loss", loss.item(), step=step)
```

`mlflow.pytorch.autolog()`를 호출하면 옵티마이저 step·learning rate·모델 그래프 등을 자동으로 기록한다. 마찬가지로 rank 0에서만 호출한다.

## TorchDistributor / `@distributed` 사용 시

함수 안에서 새로 `mlflow.start_run`을 하지 말고, **driver에서 run_id를 넘긴 뒤 child에서 `mlflow.start_run(run_id=run_id, nested=False)`로 attach**한다.

```python
with mlflow.start_run(run_name="...") as run:
    run_id = run.info.run_id
    TorchDistributor(num_processes=..., local_mode=False).run(train_fn, run_id=run_id, ...)
```

```python
def train_fn(run_id, ...):
    import os, mlflow
    if int(os.environ.get("RANK", "0")) == 0:
        mlflow.start_run(run_id=run_id)
    # ... training ...
```

## Lightning + MLflow

`MLFlowLogger`를 `Trainer(logger=...)`로 명시 주입한다.

```python
from lightning.pytorch.loggers import MLFlowLogger
logger = MLFlowLogger(experiment_name="...", run_name="...", tracking_uri="databricks")
trainer = pl.Trainer(logger=logger, ...)
```

## 모델 등록(Register)

학습 후 모델 state_dict를 저장하고 `mlflow.pytorch.log_model(..., registered_model_name="main.<schema>.<recommender>")`로 UC Model Registry에 등록한다. 이 단계는 각 셀의 `eval_and_register` 노트북에서 처리.

## 참고

- HF Transformers on Databricks: https://docs.databricks.com/aws/en/machine-learning/train-model/huggingface/
- MLflow autolog: https://mlflow.org/docs/latest/tracking/autolog.html
