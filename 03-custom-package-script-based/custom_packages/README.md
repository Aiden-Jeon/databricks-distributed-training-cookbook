# custom_packages

설치 가능한 파이썬 패키지(`recommender_pkg`). 03-custom-package-script-based의 노트북들이 wheel install 후 import합니다.

## 📂 구조

```
custom_packages/
├── pyproject.toml                                # hatchling 빌드 설정
├── src/
│   └── recommender_pkg/
│       ├── __init__.py
│       ├── model.py                              # TwoTowerMLP
│       ├── torch_distributor_trainer.py          # train_fn
│       └── lightning_trainer.py                  # RecommenderLightning, fit
└── (build artifact: dist/recommender_pkg-0.1.0-py3-none-any.whl)
```

## 🛠️ 빌드

```bash
cd 03-custom-package-script-based/custom_packages
uv build                                          # dist/*.whl 생성
```

## 📦 설치

```python
%pip install --quiet ./custom_packages/dist/recommender_pkg-0.1.0-py3-none-any.whl
dbutils.library.restartPython()
```

또는 cluster 라이브러리로 부착해서 모든 노트북이 자동 import 가능하게 합니다.

## 📝 02-script-based와의 코드 동등성

`model.py`, `torch_distributor_trainer.py`, `lightning_trainer.py`의 **함수 시그니처와 동작은 02-script-based와 동일**해야 합니다. 차이는 import 경로 (`recommender_pkg.model` vs 동일 폴더의 `model`) 뿐.
