# 00 · Foundations

본 폴더는 12개 셀을 읽기 전에 알아야 할 **공통 개념과 환경 지식**을 모아둡니다. 셀 README가 짧게 유지될 수 있도록, 반복되는 설명은 여기로 끌어냅니다.

## 📂 인덱스

| 문서 | 내용 |
|------|------|
| [recommender-baseline.md](recommender-baseline.md) | 모든 셀이 공유하는 Two-Tower MLP 모델 정의 |
| [distributed-training-concepts.md](distributed-training-concepts.md) | DDP / 임베딩 테이블 / mixed precision 개념 |
| [databricks-environments.md](databricks-environments.md) | Classic GPU vs Serverless GPU vs AI Runtime 비교 |
| [cluster-recipes.md](cluster-recipes.md) | 토폴로지·데이터 규모별 권장 인스턴스 |
| [data-loading.md](data-loading.md) | 합성 user-item interaction 데이터 생성과 로딩 패턴 |
| [mlflow-tracking.md](mlflow-tracking.md) | 분산 학습에서 MLflow autolog 사용법 |
| [uc-volumes-checkpoints.md](uc-volumes-checkpoints.md) | UC Volumes에 체크포인트 저장하는 패턴 |
| [common-pitfalls.md](common-pitfalls.md) | pickle, NCCL, OOM 등 자주 부딪히는 함정 |

## 🧭 어디서부터 읽나

- 분산 학습 처음: `distributed-training-concepts.md` → `recommender-baseline.md` → `cluster-recipes.md`
- Databricks 환경 선택부터: `databricks-environments.md` → `cluster-recipes.md`
- 모델은 알고 데이터·체크포인트만 궁금: `data-loading.md` + `uc-volumes-checkpoints.md`
- 디버깅 중: `common-pitfalls.md`

## ➡️ 다음

읽기를 마쳤다면 [루트 매트릭스](../index.html)로 돌아가 셀을 고릅니다.
