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
| `debug-` | 디버깅·함정 모음 |

### concepts-

| 문서 | 내용 |
|------|------|
| [concepts-recommender-baseline.md](concepts-recommender-baseline.md) | 모든 셀이 공유하는 Two-Tower MLP 모델 정의 |
| [concepts-distributed-training.md](concepts-distributed-training.md) | DDP·TorchDistributor `local_mode`·Lightning과의 조합·rank 간 metric 합산 |
| [concepts-torchdistributor-internals.md](concepts-torchdistributor-internals.md) | TorchDistributor 내부 동작 — Spark task, rendezvous, py4j 단절 |

### env-

| 문서 | 내용 |
|------|------|
| [env-databricks-environments.md](env-databricks-environments.md) | Classic GPU 환경, DBR 17.3 LTS ML 가정 |
| [env-cluster-recipes.md](env-cluster-recipes.md) | 토폴로지·데이터 규모별 권장 인스턴스 |
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
| [ops-resume-training.md](ops-resume-training.md) | spot interruption / Job retry에서 학습 이어받기 |

### debug-

| 문서 | 내용 |
|------|------|
| [debug-common-pitfalls.md](debug-common-pitfalls.md) | pickle, NCCL, OOM 등 자주 부딪히는 함정 + 진단 표 |

## 🧭 어디서부터 읽나

상황에 맞춰 다음 순서로 읽으면 됩니다.

- Databricks 환경 첫 셋업: `env-databricks-environments.md` → `env-cluster-recipes.md` → `env-library-management.md`
- 학습 코드 작성 전: `concepts-recommender-baseline.md` → `concepts-distributed-training.md` → `concepts-torchdistributor-internals.md`
- 데이터·체크포인트: `data-pipeline.md` → `data-uc-volumes-checkpoints.md` → `ops-resume-training.md`
- MLflow / 권한 / 인증: `ops-mlflow-tracking.md` → `env-auth.md`
- 디버깅 중: `debug-common-pitfalls.md` (상단의 진단 표부터)

## ➡️ 다음

읽기를 마쳤다면 [루트 매트릭스](../index.html)로 돌아가 셀을 고릅니다.
