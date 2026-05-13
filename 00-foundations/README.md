# 00 · Foundations

본 폴더는 12개 셀을 읽기 전에 알아 두면 좋은 **공통 개념과 환경 지식**을 모아 둡니다. 셀 README가 짧게 유지될 수 있도록 반복되는 설명을 이곳으로 끌어냈습니다.

## 📂 인덱스

파일명은 `<group>-<topic>.md` 형식이며, 그룹 의미는 다음과 같습니다.

| 그룹 | 의미 |
|------|------|
| `concepts-` | 분산 학습 핵심 개념·기준 모델·launcher 내부 동작 |
| `env-` | Databricks 런타임·클러스터·라이브러리·인증 |
| `data-` | 데이터 파이프라인·체크포인트 저장소 |
| `ops-` | 실험 추적·학습 재개 등 운영 측면 |

### concepts-

| 문서 | 내용 |
|------|------|
| [concepts-recommender-baseline.md](concepts-recommender-baseline.md) | 데이터셋·Two-Tower MLP 모델·DDP 전략·rank 간 metric 합산(AverageMeter) |
| [concepts-torchdistributor-internals.md](concepts-torchdistributor-internals.md) | TorchDistributor 내부 동작 — `local_mode`, Spark task, rendezvous |
| [concepts-torchdistributor-with-lightning.md](concepts-torchdistributor-with-lightning.md) | TorchDistributor × Lightning 조합 — driver 사용 여부, 셀 매핑, 함정 |

### env-

| 문서 | 내용 |
|------|------|
| [env-databricks-environments.md](env-databricks-environments.md) | Classic GPU 환경, DBR 17.3 LTS ML 가정, 토폴로지별 권장 인스턴스 |
| [env-library-management.md](env-library-management.md) | `%pip` vs cluster library vs wheel install, child 가시성 |
| [env-auth.md](env-auth.md) | PAT / service principal / OAuth 패턴과 권한 함정 |

### data-

| 문서 | 내용 |
|------|------|
| [data-pipeline.md](data-pipeline.md) | ML-25M 다운로드 + implicit feedback 변환 + parquet shard + DataLoader 패턴 |
| [data-uc-volumes-checkpoints.md](data-uc-volumes-checkpoints.md) | UC Volumes에 체크포인트 저장하는 패턴 |

### ops-

| 문서 | 내용 |
|------|------|
| [ops-mlflow-tracking.md](ops-mlflow-tracking.md) | 분산 학습에서 MLflow autolog 사용법 + Databricks-specific |
| [ops-uc-model-registry.md](ops-uc-model-registry.md) | 학습 후 Unity Catalog Model Registry에 등록하는 흐름 |
| [ops-resume-training.md](ops-resume-training.md) | spot interruption / Job retry에서 학습 이어받기 |

## 🧭 어디서부터 읽나

상황에 맞춰 다음 순서로 읽으면 됩니다.

- Databricks 환경 첫 셋업: `env-databricks-environments.md` → `env-library-management.md`
- 학습 코드 작성 전: `concepts-recommender-baseline.md` → `concepts-torchdistributor-internals.md` → `concepts-torchdistributor-with-lightning.md`
- 데이터·체크포인트: `data-pipeline.md` → `data-uc-volumes-checkpoints.md` → `ops-resume-training.md`
- MLflow / 권한 / 인증: `ops-mlflow-tracking.md` → `ops-uc-model-registry.md` → `env-auth.md`
- 디버깅 중: [`../docs/debug-common-pitfalls.md`](../docs/debug-common-pitfalls.md) (상단의 진단 표부터)

## ➡️ 다음

읽기를 마쳤다면 [루트 매트릭스](../index.qmd)로 돌아가 셀을 고릅니다.
