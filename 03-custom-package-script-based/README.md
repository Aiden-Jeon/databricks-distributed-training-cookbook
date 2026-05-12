# 03 · Custom-package Script-based

> 학습 로직을 **설치 가능한 파이썬 패키지**로 만들고, wheel install 후 패키지명으로 import.

## 🧭 파일 구성

| 종류 | 위치 | 역할 |
|------|------|------|
| 패키지 | [`custom_packages/`](https://github.com/Aiden-Jeon/databricks-distributed-training/tree/main/03-custom-package-script-based/custom_packages) | `pyproject.toml` + `src/` 구조. `uv build`로 wheel 생성, `%pip install`로 클러스터에 설치 |
| 노트북 | [`00-setup.ipynb`](00-setup.ipynb) | `uv build` + `%pip install dist/*.whl` + CONFIG/경로 정의 |
| 노트북 | [`01-data_prep.ipynb`](01-data_prep.ipynb) | MovieLens 25M → UC Volume parquet |
| 노트북 | [`02-launch_torch_distributor_1x1.ipynb`](02-launch_torch_distributor_1x1.ipynb) | TorchDistributor 1×1 |
| 노트북 | [`03-launch_torch_distributor_1xN.ipynb`](03-launch_torch_distributor_1xN.ipynb) | TorchDistributor 1×N |
| 노트북 | [`04-launch_torch_distributor_MxN.ipynb`](04-launch_torch_distributor_MxN.ipynb) | TorchDistributor M×N |
| 노트북 | [`05-launch_lightning_trainer_1x1.ipynb`](05-launch_lightning_trainer_1x1.ipynb) | Lightning 1×1 (driver 직접) |
| 노트북 | [`06-launch_lightning_trainer_1xN.ipynb`](06-launch_lightning_trainer_1xN.ipynb) | Lightning 1×N (TorchDistributor) |
| 노트북 | [`07-launch_lightning_trainer_MxN.ipynb`](07-launch_lightning_trainer_MxN.ipynb) | Lightning M×N |
| 노트북 | [`08-launch_accelerator_MxN.ipynb`](08-launch_accelerator_MxN.ipynb) | `accelerate launch -m recommender_pkg.torch_distributor_trainer` (subprocess.Popen, 가시 GPU 자동 감지로 1×1/1×N/M×N 전부 커버) |

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
| Accelerate | 08 (단일 노트북, 자동 감지) | 08 (동일) | 08 (동일) |

> Accelerate 행은 `sys.executable -m accelerate.commands.accelerate_cli launch -m recommender_pkg.torch_distributor_trainer` 를 `subprocess.Popen` 으로 호출합니다. 시스템 `/databricks/python3/bin/accelerate` 가 wheel 로 설치된 `recommender_pkg` 를 보지 못하므로, notebook-scoped Python(`sys.executable`)에서 모듈 모드로 띄우는 것이 필수입니다. `--num_processes` 는 생략 — 가시 GPU 수가 자동으로 채워져 한 노트북으로 1×1/1×N/M×N 모두 커버합니다.

## 📈 기대 결과

02-script-based와 동일 코드·동일 결과. 차이는 import 경로(`recommender_pkg.*` vs sibling `.py`)와 빌드 단계뿐.

| 노트북 | 토폴로지 | 학습 시간 | val/loss | GPU util |
|--------|---------|----------|----------|----------|
| 02 | 1×1 | 3~6분 | ≈ 0.45 ~ 0.55 | 60~85% |
| 03 | 1×N | 2~4분 | ≈ 0.45 ~ 0.55 | 50~80% |
| 04 | M×N | 2~3분 | ≈ 0.45 ~ 0.55 | 40~70% |
| 05 | Lightning 1×1 | 3~6분 | ≈ 0.45 ~ 0.55 | 60~85% |
| 06 | Lightning 1×N | 2~4분 | ≈ 0.45 ~ 0.55 | 50~80% |
| 07 | Lightning M×N | 2~3분 | ≈ 0.45 ~ 0.55 | 40~70% |
| 08 | Accelerate (auto) | 토폴로지에 따라 위와 동일 | 동일 | 동일 |

추가 오버헤드: `uv build` (00-setup 안에서 ~수 초) + `%pip install ./dist/*.whl` (~10초). 한 번 빌드/설치하면 후속 노트북에서 즉시 import 가능.

08 (Accelerate) 가 02-script-based의 08과 다른 점: 시스템 `accelerate` 가 wheel을 못 보므로 `sys.executable -m accelerate.commands.accelerate_cli launch -m recommender_pkg.torch_distributor_trainer` 모듈 모드 호출이 필수 ([`env-library-management.md`](../00-foundations/env-library-management.md) §"함정 4").

확장 시나리오 (본 쿡북 밖):
- wheel을 PyPI / 내부 artifact registry에 publish → 여러 cluster·Job에서 동일 버전 import
- CI에서 `uv build` + 테스트 자동화

## ➡️ 다음

foundations로 돌아가기: [`../00-foundations/`](../00-foundations/README.md)
