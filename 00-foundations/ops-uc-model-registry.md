# Unity Catalog Model Registry

학습이 끝난 모델을 Unity Catalog Model Registry에 등록하는 흐름을 정리합니다. MLflow tracking 자체와 LoggedModel 로깅 방식은 [`ops-mlflow-tracking.md`](ops-mlflow-tracking.md), 권한 진단 표는 [`env-auth.md`](env-auth.md)를 참조하세요.

## 핵심 한 줄

```python
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(model_uri, "main.distributed_cookbook.two_tower_mlp")
```

`mlflow.set_registry_uri("databricks-uc")`가 UC Model Registry로 라우팅하는 핵심 호출입니다. DBR 17.3 LTS ML은 default가 UC라 호출 없이도 동작하지만, 명시는 워크스페이스 registry로 잘못 등록되는 사고를 막아주는 안전장치입니다. 워크스페이스 registry로 등록하려면 `"databricks"`로 지정합니다.

## 등록 흐름

등록에는 시그니처가 반드시 필요합니다. 시그니처는 학습 중 `mlflow.pytorch.log_model(..., signature=...)`로 함께 기록한 것을 그대로 사용합니다(예제는 [`ops-mlflow-tracking.md`](ops-mlflow-tracking.md)의 LoggedModel 섹션).

```python
import mlflow

mlflow.set_registry_uri("databricks-uc")

uc_model_name = "main.distributed_cookbook.two_tower_mlp"
model_uri = f"runs:/{run_id}/model"           # 또는 best LoggedModel의 model_uri

registered = mlflow.register_model(model_uri, uc_model_name)
print(f"Registered {uc_model_name} v{registered.version}")
```

## Best LoggedModel 선택

MLflow 3.0+에서는 한 run에 epoch별로 여러 `LoggedModel`이 있습니다. 학습 종료 후 메트릭과 묶어 best를 골라 등록합니다.

```python
from mlflow import MlflowClient

client = MlflowClient()
run = client.get_run(run_id)
model_outputs = run.outputs.model_outputs    # 이 run에서 로깅된 모든 LoggedModel
# ... 메트릭과 묶어 정렬 후 best 선택 ...
```

## 권한

가장 자주 막히는 부분입니다. 필요한 grant 요약은 다음과 같습니다.

| 권한 | 대상 |
|------|------|
| `USE CATALOG` | catalog (예: `main`) |
| `USE SCHEMA` | schema (예: `main.distributed_cookbook`) |
| `CREATE MODEL` | schema |

흔한 에러는 `PERMISSION_DENIED: User does not have CREATE MODEL on schema 'main.distributed_cookbook'`이며, 해결은 `` GRANT CREATE MODEL ON SCHEMA main.distributed_cookbook TO `<user-or-sp>` `` 실행입니다. PAT/사용자 vs SP 차이와 자세한 진단 표는 [`env-auth.md`](env-auth.md)의 "권한 함정 — UC Model Registry" 섹션을 참조하세요.

## 참고

- [Unity Catalog Model Registry](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/index)
- [MLflow Model Registry API](https://mlflow.org/docs/latest/model-registry.html)
