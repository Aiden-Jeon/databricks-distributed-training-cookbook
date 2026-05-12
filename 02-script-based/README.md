# 02 · Script-based

> 학습 로직을 `.py` 모듈로 분리. 노트북은 driver(파라미터 구성·실행·결과)만 담당.

## 🧭 파일 구성

| 종류 | 파일 | 역할 |
|------|------|------|
| 모듈 | [`model.py`](https://github.com/Aiden-Jeon/databricks-distributed-training/blob/main/02-script-based/model.py) | `TwoTowerMLP` + `EarlyStopping`. 모든 trainer가 import |
| 모듈 | [`torch_distributor_trainer.py`](https://github.com/Aiden-Jeon/databricks-distributed-training/blob/main/02-script-based/torch_distributor_trainer.py) | TorchDistributor / accelerate launch가 사용하는 `train_fn(...)` (raw `dist.*` DDP). `__main__`은 accelerate launch 진입점 |
| 모듈 | [`lightning_trainer.py`](https://github.com/Aiden-Jeon/databricks-distributed-training/blob/main/02-script-based/lightning_trainer.py) | Lightning `LitModule` + `DataModule` + `fit(...)` |
| 노트북 | [`00-setup.ipynb`](00-setup.ipynb) | 환경 설정 (UC 경로, CONFIG, MLflow experiment, DB_HOST/DB_TOKEN) |
| 노트북 | [`01-data_prep.ipynb`](01-data_prep.ipynb) | 데이터 준비 (01-row와 동일) |
| 노트북 | [`02-launch_torch_distributor_1x1.ipynb`](02-launch_torch_distributor_1x1.ipynb) | TorchDistributor 1×1 |
| 노트북 | [`03-launch_torch_distributor_1xN.ipynb`](03-launch_torch_distributor_1xN.ipynb) | TorchDistributor 1×N |
| 노트북 | [`04-launch_torch_distributor_MxN.ipynb`](04-launch_torch_distributor_MxN.ipynb) | TorchDistributor M×N |
| 노트북 | [`05-launch_lightning_trainer_1x1.ipynb`](05-launch_lightning_trainer_1x1.ipynb) | Lightning 1×1 (driver 직접 `fit`) |
| 노트북 | [`06-launch_lightning_trainer_1xN.ipynb`](06-launch_lightning_trainer_1xN.ipynb) | Lightning 1×N (TorchDistributor + `Trainer`) |
| 노트북 | [`07-launch_lightning_trainer_MxN.ipynb`](07-launch_lightning_trainer_MxN.ipynb) | Lightning M×N |
| 노트북 | [`08-launch_accelerator_MxN.ipynb`](08-launch_accelerator_MxN.ipynb) | `accelerate launch <script>` (subprocess.Popen, 가시 GPU 자동 감지로 1×1/1×N/M×N 전부 커버) |

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

> 패키지로 묶어 `%pip install` 하는 형태가 필요하면 [`03-custom-package-script-based/`](../03-custom-package-script-based/README.md).

## 🔀 매트릭스

같은 `train_fn`(또는 `fit`)을 세 launcher × 세 토폴로지로 호출.

| | TorchDistributor | Lightning | Accelerate |
|----|---|---|---|
| **1×1** | 02 | 05 (driver 직접) | 08 |
| **1×N** | 03 | 06 | 08 |
| **M×N** | 04 | 07 | 08 |

> 한 노트북 세션에서 `TorchDistributor.run`을 연속 호출하면 driver py4j callback channel이 단절되는 현상이 관찰돼 (DBR 17.3 LTS ML, g5.12xlarge), launcher × topology 별로 노트북을 분리했습니다. 토폴로지를 비교할 때는 같은 클러스터에서 노트북을 차례로 detach → 다음 노트북 attach 하여 실행하세요.
>
> Accelerate 행은 한 개 노트북(08)으로 1×1/1×N/M×N 을 모두 커버합니다. `accelerate launch` 가 가시 GPU 수를 자동 감지하므로 클러스터 사양을 바꾸기만 하면 동일 노트북이 그대로 동작합니다. 호출은 `subprocess.Popen` + `shlex.split` 패턴 — 자식 프로세스에 `DATABRICKS_HOST/TOKEN` 명시 주입, Py4J keepalive 스레드로 장시간 학습 중 gateway 단절 방지, non-zero return 시 `dbutils.notebook.exit` 로 실패 종료.

## 🖥️ 클러스터 세팅

| 토폴로지 | 노트북 | Single node 토글 | Workers |
|----------|--------|-----------------|---------|
| 1×1 | 02, 05, 08 | ON | 0 |
| 1×N | 03, 06, 08 | ON | 0 |
| M×N | 04, 07, 08 | OFF | M (예: 1~3) |

driver/worker는 모두 `g5.12xlarge` (4× A10G) 권장. Autoscaling 항상 OFF. 1×1은 `g5.2xlarge` (1× A10G)로도 충분. 상세는 [`00-foundations/cluster-recipes.md`](../00-foundations/cluster-recipes.md).

## 📈 기대 결과

01-notebook-based와 동일 코드·동일 데이터·동일 launcher 설정. 차이는 import 방식뿐이므로 결과도 동일해야 합니다.

| 노트북 | 토폴로지 | 학습 시간 | val/loss | GPU util |
|--------|---------|----------|----------|----------|
| 02 | 1×1 | 3~6분 | ≈ 0.45 ~ 0.55 | 60~85% |
| 03 | 1×N | 2~4분 | ≈ 0.45 ~ 0.55 | 50~80% |
| 04 | M×N | 2~3분 | ≈ 0.45 ~ 0.55 | 40~70% |
| 05 | Lightning 1×1 | 3~6분 | ≈ 0.45 ~ 0.55 | 60~85% |
| 06 | Lightning 1×N | 2~4분 | ≈ 0.45 ~ 0.55 | 50~80% |
| 07 | Lightning M×N | 2~3분 | ≈ 0.45 ~ 0.55 | 40~70% |
| 08 | Accelerate (auto) | 토폴로지에 따라 위와 동일 | 동일 | 동일 |

스크립트 분리 자체로 인한 추가 오버헤드는 없습니다 — `import` 한 함수가 같은 Python interpreter에서 실행되므로 01과 동일. 차이가 크다면:
- `sys.path` 보강이 child에서 누락되어 import 실패 → Spark UI executor log 확인 ([`common-pitfalls.md §11`](../00-foundations/common-pitfalls.md))
- 노트북 분리 후 cluster를 재시작했다면 `%pip install` 휘발 — `00-setup` 재실행 필요 ([`library-management.md`](../00-foundations/library-management.md))

## ➡️ 다음

설치 가능 wheel로 패키징한 버전: [`03-custom-package-script-based/`](../03-custom-package-script-based/README.md)
