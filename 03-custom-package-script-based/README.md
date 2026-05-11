# 03 · Custom-package Script-based

> 학습 로직을 **설치 가능한 파이썬 패키지**로 만들고, wheel install 후 패키지명으로 import.

## 🧭 파일 구성

| 종류 | 위치 | 역할 |
|------|------|------|
| 패키지 | [`custom_packages/`](custom_packages/) | `pyproject.toml` + `src/` 구조. `uv build`로 wheel 생성, `%pip install`로 클러스터에 설치 |
| 노트북 | [`00-setup.ipynb`](00-setup.ipynb) | `uv build` + `%pip install dist/*.whl` + CONFIG/경로 정의 |
| 노트북 | [`01-data_prep.ipynb`](01-data_prep.ipynb) | MovieLens 25M → UC Volume parquet |
| 노트북 | [`02-launch_torch_distributor_1x1.ipynb`](02-launch_torch_distributor_1x1.ipynb) | TorchDistributor 1×1 |
| 노트북 | [`03-launch_torch_distributor_1xN.ipynb`](03-launch_torch_distributor_1xN.ipynb) | TorchDistributor 1×N |
| 노트북 | [`04-launch_torch_distributor_MxN.ipynb`](04-launch_torch_distributor_MxN.ipynb) | TorchDistributor M×N |
| 노트북 | [`05-launch_lightning_trainer_1x1.ipynb`](05-launch_lightning_trainer_1x1.ipynb) | Lightning 1×1 (driver 직접) |
| 노트북 | [`06-launch_lightning_trainer_1xN.ipynb`](06-launch_lightning_trainer_1xN.ipynb) | Lightning 1×N (TorchDistributor) |
| 노트북 | [`07-launch_lightning_trainer_MxN.ipynb`](07-launch_lightning_trainer_MxN.ipynb) | Lightning M×N |
| 노트북 | [`08-launch_accelerator_MxN.ipynb`](08-launch_accelerator_MxN.ipynb) | Accelerator API M×N (TorchDistributor dispatcher) |

토폴로지(1×1 / 1×N / M×N)는 02-script-based와 동일하게 노트북 단위로 분리했습니다. 한 세션에서 `TorchDistributor.run`을 연속 호출하면 py4j callback channel이 단절되는 현상을 피하기 위함입니다.

## 🔌 Import 방식 — `02-script-based`와의 차이

| 항목 | `02-script-based` | `03-custom-package-script-based` (이 폴더) |
|------|-------------------|-----------------------------------------|
| 코드 위치 | 노트북과 같은 폴더의 `.py` | `custom_packages/src/recommender_pkg/` 패키지 |
| Import 방식 | `from torch_distributor_trainer import train_fn` (sys.path 보강 필요) | `from recommender_pkg.torch_distributor_trainer import train_fn` (wheel 설치 후) |
| 빌드 | 없음 | `uv build` → `dist/*.whl` |
| 배포 | Databricks Repos에 폴더째 push | wheel을 `%pip install` 또는 cluster 라이브러리로 부착 |
| 적합한 경우 | PoC, 단일 프로젝트 | 여러 노트북/job에서 재사용, 버전 관리, CI/CD |

## 🛠️ 빌드/설치 흐름

```python
# 00-setup.ipynb에서 자동 수행
%pip install --quiet uv "lightning==2.5.1" "nvidia-ml-py"
%restart_python

import subprocess
subprocess.run(["uv", "build"], cwd=f"{NOTEBOOK_DIR}/custom_packages", check=True)

%pip install --quiet ./custom_packages/dist/recommender_pkg-0.1.0-py3-none-any.whl
%restart_python
```

## 🔀 매트릭스

`02-script-based`와 동일. 차이는 import 방식만.

| launcher | 1×1 GPU | 1×N GPU | M×N GPU |
|----------|---------|---------|---------|
| TorchDistributor | `TorchDistributor(num_processes=1, local_mode=True).run(td_train_fn, ...)` | `TorchDistributor(num_processes=N, local_mode=True).run(...)` | `TorchDistributor(num_processes=M*N, local_mode=False).run(...)` |
| Lightning | `fit(devices=1, num_nodes=1)` 직접 호출 | `TorchDistributor(num_processes=N, local_mode=True).run(td_lit_fit, devices=N, num_nodes=1)` | `TorchDistributor(num_processes=M*N, local_mode=False).run(td_lit_fit, devices=N, num_nodes=M)` |
| Accelerate | (생략 — TD 1×1로 대체) | (생략 — TD 1×N으로 대체) | `TorchDistributor(num_processes=M*N, local_mode=False).run(train_fn_acc, ...)` (Accelerator API) |

> Accelerate 1×1·1×N은 02-script-based와 같은 `accelerate launch ./trainer.py` 패턴이 가능하지만, 패키지 설치 시 `sys.executable -m accelerate.commands.accelerate_cli launch -m recommender_pkg.torch_distributor_trainer` 형태로 호출해야 notebook-scoped Python env의 `recommender_pkg`이 보입니다. 본 행에서는 단순성을 위해 1×1·1×N은 TorchDistributor·Lightning 노트북으로 대체하고, Accelerate는 **M×N(TorchDistributor dispatcher + Accelerator API)** 한 가지만 다룹니다.

## ➡️ 다음

foundations로 돌아가기: [`../00-foundations/`](../00-foundations/)
