# 03 · Custom-package Script-based

> 학습 로직을 **설치 가능한 파이썬 패키지**로 만들고, wheel install 후 패키지명으로 import.

## 🧭 파일 구성

| 종류 | 위치 | 역할 |
|------|------|------|
| 패키지 | [`custom_packages/`](custom_packages/) | `pyproject.toml` + `src/` 구조. `uv build`로 wheel 생성, `%pip install`로 클러스터에 설치 |
| 노트북 | [`00-setup.ipynb`](00-setup.ipynb) | `uv build` + `%pip install dist/*.whl` |
| 노트북 | [`01-data_prep.ipynb`](01-data_prep.ipynb) | 데이터 준비 (패키지의 데이터 유틸 import) |
| 노트북 | [`02-launch_torch_distributor.ipynb`](02-launch_torch_distributor.ipynb) | `from custom_packages.trainer import train_fn` 후 TorchDistributor 호출 |
| 노트북 | [`03-launch_lightning_trainer.ipynb`](03-launch_lightning_trainer.ipynb) | Lightning Trainer 호출 |
| 노트북 | [`04-launch_accelerator.ipynb`](04-launch_accelerator.ipynb) | `accelerate launch` 호출 |

## 🔌 Import 방식 — `02-script-based`와의 차이

| 항목 | `02-script-based` | `03-custom-package-script-based` (이 폴더) |
|------|-------------------|-----------------------------------------|
| 코드 위치 | 노트북과 같은 폴더의 `.py` | `custom_packages/src/<name>/` 패키지 |
| Import 방식 | 상대 경로 / `%run` | `from <package_name> import ...` (wheel 설치 후) |
| 빌드 | 없음 | `uv build` → `dist/*.whl` |
| 배포 | Databricks Repos에 폴더째 push | wheel을 `%pip install` 또는 cluster 라이브러리로 부착 |
| 적합한 경우 | PoC, 단일 프로젝트 | 여러 노트북/job에서 재사용, 버전 관리, CI/CD |

## 🛠️ 빌드/설치 흐름

```bash
# 로컬 또는 Databricks
cd custom_packages
uv build                                       # dist/<name>-<ver>-py3-none-any.whl 생성
```

```python
# 노트북 cell
%pip install --quiet ./custom_packages/dist/<name>-<ver>-py3-none-any.whl
dbutils.library.restartPython()
```

## 🔀 매트릭스

`02-script-based`와 동일. 차이는 import 방식만.

## ➡️ 다음

foundations로 돌아가기: [`../00-foundations/`](../00-foundations/)
