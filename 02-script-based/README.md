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
| 노트북 | [`02-launch_torch_distributor_single_node.ipynb`](02-launch_torch_distributor_single_node.ipynb) | TorchDistributor 1×1 / 1×N 섹션 |
| 노트북 | [`03-launch_torch_distributor_multi_node.ipynb`](03-launch_torch_distributor_multi_node.ipynb) | TorchDistributor M×N |
| 노트북 | [`04-launch_lightning_trainer_single_node.ipynb`](04-launch_lightning_trainer_single_node.ipynb) | Lightning 1×1 / 1×N 섹션 |
| 노트북 | [`05-launch_lightning_trainer_multi_node.ipynb`](05-launch_lightning_trainer_multi_node.ipynb) | Lightning M×N |
| 노트북 | [`06-launch_accelerator_single_node.ipynb`](06-launch_accelerator_single_node.ipynb) | `accelerate launch` 1×1 / 1×N (subprocess) |
| 노트북 | [`07-launch_accelerator_multi_node.ipynb`](07-launch_accelerator_multi_node.ipynb) | Accelerator API M×N (TorchDistributor dispatcher) |

## 🔌 Import 방식

노트북이 위치한 Workspace 폴더를 `sys.path`에 직접 삽입한 뒤 sibling `.py` 모듈을 import합니다. Multi-node에서는 worker 프로세스가 다른 노드에서 fresh Python으로 시작하므로 **`script_dir`을 child 함수의 인자로 명시 전달**하고, child 함수가 내부에서 `sys.path.insert(0, script_dir)`을 호출합니다.

```python
NOTEBOOK_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
SCRIPT_DIR = "/Workspace" + os.path.dirname(NOTEBOOK_PATH)
sys.path.insert(0, SCRIPT_DIR)
from torch_distributor_trainer import train_fn

TorchDistributor(...).run(train_fn, ..., script_dir=SCRIPT_DIR)
```

> 패키지로 묶어 `%pip install` 하는 형태가 필요하면 [`03-custom-package-script-based/`](../03-custom-package-script-based/).

## 🔀 매트릭스

같은 `train_fn`(또는 `fit`)을 세 launcher × 세 토폴로지로 호출.

| | TorchDistributor | Lightning | Accelerate |
|----|---|---|---|
| **1×1** | 02: `local_mode=True, num_processes=1` | 04: `Trainer(devices=1)` 직접 호출 | 06: `accelerate launch --num_processes 1` |
| **1×N** | 02: `local_mode=True, num_processes=N` | 04: TorchDistributor + `Trainer(devices=N, strategy='ddp')` | 06: `accelerate launch --multi_gpu --num_processes N` |
| **M×N** | 03: `local_mode=False, num_processes=M*N` | 05: TorchDistributor + `Trainer(devices=N, num_nodes=M, strategy='ddp')` | 07: TorchDistributor + `Accelerator()` API |

> 06의 `accelerate launch`는 single-node 전용입니다. native `accelerate launch --multi_gpu --num_machines M --machine_rank R`은 노드별 dispatch를 요구하지만 Databricks에서는 단일 driver 노트북에서 이를 직접 띄울 수 없습니다 — 07이 그 우회 패턴(TorchDistributor를 dispatcher로 사용 + Accelerator API)을 보여줍니다.

## 🖥️ 클러스터 세팅

01-row와 동일 매핑. 02·04·06(single-node)은 같은 single-node 클러스터에서 모두 실행 가능. 03·05·07(multi-node)은 multi-node 클러스터에서.

| 항목 | Single-node (02, 04, 06) | Multi-node (03, 05, 07) |
|------|--------------------------|--------------------------|
| Single node 토글 | **ON** | **OFF** |
| Driver type | `g5.12xlarge` (4× A10G) | `g5.12xlarge` (4× A10G) |
| Worker type | — | `g5.12xlarge` (4× A10G) |
| Workers | 0 | M (예: 1~3) |
| Autoscaling | off | **off (필수)** |

상세는 [`00-foundations/cluster-recipes.md`](../00-foundations/cluster-recipes.md).

## ➡️ 다음

설치 가능 wheel로 패키징한 버전: [`03-custom-package-script-based/`](../03-custom-package-script-based/)
