# 02 · Script-based

> 학습 로직을 `.py` 모듈로 분리. 노트북은 driver(파라미터 구성·실행·결과)만.

## 🧭 파일 구성

| 종류 | 파일 | 역할 |
|------|------|------|
| 모듈 | [`model.py`](model.py) | Two-Tower MLP 정의. 모든 trainer가 import |
| 모듈 | [`torch_distributor_trainer.py`](torch_distributor_trainer.py) | TorchDistributor와 HF Accelerate가 entry point로 사용하는 학습 함수 |
| 모듈 | [`lightning_trainer.py`](lightning_trainer.py) | Lightning `LightningModule` + `Trainer` 호출 |
| 노트북 | [`00-setup.ipynb`](00-setup.ipynb) | 환경 설정 |
| 노트북 | [`01-data_prep.ipynb`](01-data_prep.ipynb) | 데이터 준비 |
| 노트북 | [`02-launch_torch_distributor.ipynb`](02-launch_torch_distributor.ipynb) | TorchDistributor로 `torch_distributor_trainer.py` 실행 |
| 노트북 | [`03-launch_lightning_trainer.ipynb`](03-launch_lightning_trainer.ipynb) | Lightning `Trainer`로 학습 |
| 노트북 | [`04-launch_accelerator.ipynb`](04-launch_accelerator.ipynb) | `accelerate launch torch_distributor_trainer.py` 실행 |

## 🔌 Import 방식

같은 디렉터리에서 상대 경로로 import. Databricks Repos에서는 `sys.path`에 폴더가 들어 있어 그대로 동작.

```python
%run ./torch_distributor_trainer.py    # Databricks magic
# 또는
from torch_distributor_trainer import train_fn
```

> 패키지로 묶어 설치하는 형태가 필요하면 [`03-custom-package-script-based/`](../03-custom-package-script-based/) 참고.

## 🔀 매트릭스

같은 학습 함수(`train_fn`)를 세 launcher × 세 토폴로지로 호출.

| | TorchDistributor (`02`) | Lightning (`03`) | Accelerate (`04`) |
|----|---|---|---|
| **1×1** | `local_mode=True, num_processes=1` | `Trainer(devices=1)` | `accelerate launch --num_processes 1 ...` |
| **1×N** | `local_mode=True, num_processes=N` | `Trainer(devices=N)` | `accelerate launch --num_processes N ...` |
| **M×N** | `local_mode=False, num_processes=M*N` | `Trainer(devices=N, num_nodes=M)` | `accelerate launch --multi_gpu --num_machines M ...` |

## ➡️ 다음

패키지화 버전: [`03-custom-package-script-based/`](../03-custom-package-script-based/)
