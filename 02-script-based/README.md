# 02 · Script-based

> 학습 로직을 `.py` 모듈로 분리. 노트북은 driver(파라미터 구성·실행·결과)만 담당.

## 🧭 파일 구성

| 종류 | 파일 | 역할 |
|------|------|------|
| 모듈 | [`model.py`](model.py) | `TwoTowerMLP` + `EarlyStopping`. 모든 trainer가 import |
| 모듈 | [`torch_distributor_trainer.py`](torch_distributor_trainer.py) | TorchDistributor / accelerate launch가 사용하는 `train_fn(...)` (raw `dist.*` DDP). `__main__`은 accelerate launch 진입점 |
| 모듈 | [`lightning_trainer.py`](lightning_trainer.py) | Lightning `LitModule` + `DataModule` + `fit(...)` |
| 노트북 | [`00-setup.ipynb`](00-setup.ipynb) | 환경 설정 (UC 경로, CONFIG, MLflow experiment, DB_HOST/DB_TOKEN) |
| 노트북 | [`01-data_prep.ipynb`](01-data_prep.ipynb) | 데이터 준비 (01-row와 동일) |
| 노트북 | [`02-launch_torch_distributor_1x1.ipynb`](02-launch_torch_distributor_1x1.ipynb) | TorchDistributor 1×1 |
| 노트북 | [`03-launch_torch_distributor_1xN.ipynb`](03-launch_torch_distributor_1xN.ipynb) | TorchDistributor 1×N |
| 노트북 | [`04-launch_torch_distributor_MxN.ipynb`](04-launch_torch_distributor_MxN.ipynb) | TorchDistributor M×N |
| 노트북 | [`05-launch_lightning_trainer_1x1.ipynb`](05-launch_lightning_trainer_1x1.ipynb) | Lightning 1×1 (driver 직접 `fit`) |
| 노트북 | [`06-launch_lightning_trainer_1xN.ipynb`](06-launch_lightning_trainer_1xN.ipynb) | Lightning 1×N (TorchDistributor + `Trainer`) |
| 노트북 | [`07-launch_lightning_trainer_MxN.ipynb`](07-launch_lightning_trainer_MxN.ipynb) | Lightning M×N |
| 노트북 | [`08-launch_accelerator_1x1.ipynb`](08-launch_accelerator_1x1.ipynb) | `accelerate launch --num_processes 1` (subprocess) |
| 노트북 | [`09-launch_accelerator_1xN.ipynb`](09-launch_accelerator_1xN.ipynb) | `accelerate launch --multi_gpu --num_processes N` (subprocess) |
| 노트북 | [`10-launch_accelerator_MxN.ipynb`](10-launch_accelerator_MxN.ipynb) | Accelerator API M×N (TorchDistributor dispatcher) |

## 🔌 Import 방식

노트북이 위치한 Workspace 폴더를 `sys.path`에 직접 삽입한 뒤 sibling `.py` 모듈을 import합니다.

TorchDistributor `.run`에 module-level 함수를 직접 넘기면 cloudpickle이 module reference로 직렬화하고, child 프로세스의 fresh `sys.path`에는 SCRIPT_DIR이 없어 import가 실패합니다. 이를 회피하기 위해 노트북에서 **inline thin wrapper**를 정의해 by-value pickling을 받게 하고, wrapper 내부에서 `sys.path` 보강 후 lazy import합니다.

```python
NOTEBOOK_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
SCRIPT_DIR = "/Workspace" + os.path.dirname(NOTEBOOK_PATH)
sys.path.insert(0, SCRIPT_DIR)

def td_train_fn(**kwargs):
    import sys
    sd = kwargs.get("script_dir")
    if sd and sd not in sys.path:
        sys.path.insert(0, sd)
    from torch_distributor_trainer import train_fn
    return train_fn(**kwargs)

TorchDistributor(...).run(td_train_fn, ..., script_dir=SCRIPT_DIR)
```

> 패키지로 묶어 `%pip install` 하는 형태가 필요하면 [`03-custom-package-script-based/`](../03-custom-package-script-based/).

## 🔀 매트릭스

같은 `train_fn`(또는 `fit`)을 세 launcher × 세 토폴로지로 호출.

| | TorchDistributor | Lightning | Accelerate |
|----|---|---|---|
| **1×1** | 02 | 05 (driver 직접) | 08 |
| **1×N** | 03 | 06 | 09 |
| **M×N** | 04 | 07 | 10 |

> 한 노트북 세션에서 `TorchDistributor.run`을 연속 호출하면 driver py4j callback channel이 단절되는 현상이 관찰돼 (DBR 17.3 LTS ML, g5.12xlarge), launcher × topology 별로 노트북을 분리했습니다. 토폴로지를 비교할 때는 같은 클러스터에서 노트북을 차례로 detach → 다음 노트북 attach 하여 실행하세요.
>
> 10 (Accelerate M×N)은 native `accelerate launch --multi_gpu --num_machines M --machine_rank R`이 노드별 dispatch를 요구하지만 Databricks 단일 driver 노트북에서 직접 띄울 수 없어, TorchDistributor를 dispatcher로 쓰고 child에서 `Accelerator()` API를 사용합니다.

## 🖥️ 클러스터 세팅

| 토폴로지 | 노트북 | Single node 토글 | Workers |
|----------|--------|-----------------|---------|
| 1×1 | 02, 05, 08 | ON | 0 |
| 1×N | 03, 06, 09 | ON | 0 |
| M×N | 04, 07, 10 | OFF | M (예: 1~3) |

driver/worker는 모두 `g5.12xlarge` (4× A10G) 권장. Autoscaling 항상 OFF. 1×1은 `g5.2xlarge` (1× A10G)로도 충분. 상세는 [`00-foundations/cluster-recipes.md`](../00-foundations/cluster-recipes.md).

## ➡️ 다음

설치 가능 wheel로 패키징한 버전: [`03-custom-package-script-based/`](../03-custom-package-script-based/)
