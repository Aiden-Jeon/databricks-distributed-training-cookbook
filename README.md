# Databricks Distributed Training Cookbook

Databricks 환경에서 **추천 모델(Two-Tower MLP)**을 분산학습하는 패턴을 **코드 조직 방식 × 학습 launcher** 매트릭스로 정리한 쿡북입니다.

## 🎯 누구를 위한 쿡북인가

다음 독자를 염두에 두고 작성했습니다.

- PyTorch 추천 모델을 처음 Databricks에서 분산학습하려는 ML 엔지니어
- 노트북 PoC → 스크립트 분리 → 패키지화 경로를 한 곳에서 보고 싶은 팀
- 같은 모델을 TorchDistributor / Lightning / Accelerate 중 어느 launcher로 돌릴지 비교하려는 팀

## 🧭 매트릭스

각 셀이 어떤 노트북에 매핑되는지 한눈에 보여 줍니다.

|                  | TorchDistributor | PyTorch Lightning | HF Accelerate |
|------------------|------------------|-------------------|---------------|
| **01-notebook-based** | [`02-train_torch_distributor_1x1_1xN.ipynb`](01-notebook-based/02-train_torch_distributor_1x1_1xN.ipynb) (1×1, 1×N)<br>[`03-train_torch_distributor_MxN.ipynb`](01-notebook-based/03-train_torch_distributor_MxN.ipynb) (M×N) | [`04-train_pytorch_lightning_1x1_1xN.ipynb`](01-notebook-based/04-train_pytorch_lightning_1x1_1xN.ipynb) (1×1, 1×N)<br>[`05-train_pytorch_lightning_MxN.ipynb`](01-notebook-based/05-train_pytorch_lightning_MxN.ipynb) (M×N) | — (CLI 기반이라 N/A) |
| **02-script-based** | 02 (1×1) · 03 (1×N) · 04 (M×N) | 05 (1×1) · 06 (1×N) · 07 (M×N) | 08 (1×1 / 1×N / M×N) |
| **03-custom-package-script-based** | 02 (1×1) · 03 (1×N) · 04 (M×N) | 05 (1×1) · 06 (1×N) · 07 (M×N) | 08 (1×1 / 1×N / M×N) |

02·03 행은 launcher × topology별로 노트북을 분리했습니다. 한 노트북에서 `TorchDistributor.run`을 연속 호출하면 py4j callback이 단절되는 현상을 피하기 위함입니다. Accelerate는 단일 노트북에서 가시 GPU 수를 자동 감지해 1×1/1×N/M×N을 모두 다루며, 실행은 `subprocess.Popen` 패턴을 따릅니다. 자세한 노트북 목록은 [`02-script-based/README.md`](02-script-based/README.md), [`03-custom-package-script-based/README.md`](03-custom-package-script-based/README.md)을 참고하세요.

### 행: 코드 조직 방식

행별 차별점은 다음과 같습니다.

| 행 | 차별점 |
|----|--------|
| 01-notebook-based | 학습 코드를 노트북 셀 안에 작성. 가장 빠른 PoC 경로. |
| 02-script-based | 같은 디렉터리에 `.py` 모듈(model/trainer)을 두고 노트북은 driver 역할. 상대 경로 import. |
| 03-custom-package-script-based | 학습 코드를 `pyproject.toml` 기반 설치 가능 패키지로 분리. `uv build` → wheel install 후 패키지명으로 import. 여러 노트북/job 재사용, CI/CD에 적합. |

### 열: 학습 launcher

launcher별 진입 API는 다음과 같이 달라집니다.

| 열 | API |
|----|-----|
| TorchDistributor | `TorchDistributor(num_processes, local_mode, use_gpu).run(fn, ...)` |
| PyTorch Lightning | `Trainer(accelerator="gpu", devices=N, num_nodes=M, strategy="ddp").fit(...)` |
| HF Accelerate | `%sh accelerate launch --config_file <cfg>.yaml train.py` |

### 토폴로지 (노트북 내부 섹션)

> **표기 규칙**: `M × N`에서 **M은 노드 수**, **N은 노드당 GPU 수**입니다. 학습 process(rank)의 총 개수는 `world_size = M × N`입니다.

| 섹션 | 의미 | world_size | 권장 환경 |
|------|------|------------|----------|
| 1×1 GPU | 1 노드, 노드당 1 GPU | 1 | Classic GPU 단일 인스턴스 또는 Serverless GPU |
| 1×N GPU | 1 노드, 노드당 N GPU | N | Classic 다중 GPU 인스턴스 (g5.12xlarge 등) |
| M×N GPU | M 노드, 노드당 N GPU | M × N | Classic 멀티 노드 (TorchDistributor 백엔드) |

01-notebook-based 행은 `1x1_1xN` 노트북에서 1×1·1×N을 한 노트북의 섹션으로 함께 다루고, `MxN` 노트북에서 M×N을 따로 다룹니다. 02·03 행은 launcher × topology별로 노트북이 분리되어 파일명에 `1x1` / `1xN` / `MxN`이 들어갑니다.

## 🧱 공통 스택

쿡북 전반에서 사용하는 기술 스택은 다음과 같습니다.

- 모델: Two-Tower MLP 추천 모델 ([00-foundations/concepts-recommender-baseline.md](00-foundations/concepts-recommender-baseline.md))
- 학습 라이브러리: `torch`, `accelerate`, `pytorch-lightning`
- 병렬화: PyTorch **DDP** (FSDP/DeepSpeed는 다루지 않음)
- 데이터: MovieLens 25M (implicit feedback 변환) → UC Volume parquet shard → torch DataLoader
- 체크포인트: Unity Catalog Volumes
- 실험 추적: MLflow (3.0+ 패턴)
- Multi-node launcher: TorchDistributor

## 🗺️ 시작 가이드

다음 순서로 따라가는 것을 권장합니다.

1. 처음이라면 [`00-foundations/`](00-foundations/README.md)부터 읽습니다. 기준 모델 정의, Classic GPU 환경, 클러스터 권장 사양, 데이터 로딩, MLflow, UC Volumes, TorchDistributor 내부 동작, 라이브러리·인증 관리, 흔한 함정을 모두 다룹니다.
2. 매트릭스에서 자신의 시나리오에 가장 가까운 셀(노트북)을 엽니다.
3. 해당 행(`01-notebook-based` / `02-script-based` / `03-custom-package-script-based`)의 README "실행 순서"를 따릅니다.
4. 토폴로지 섹션을 1×1 → 1×N → M×N 순으로 확장합니다.

### 🧗 추천 학습 동선

분산 학습 자체는 익숙하지만 Databricks 위에서 처음 돌려 보는 경우, 다음 순서로 학습 시간을 최소화하면서 점진적으로 확장할 수 있습니다.

| 단계 | 셀 | 목표 | 검증 포인트 |
|------|----|----|------------|
| 1 | `01-notebook-based/00-setup` + `01-data_prep` | UC Volume에 ML-25M 준비 | `_meta.json` 의 n_users/n_items, train shard 개수 |
| 2 | `01-notebook-based/02` (1×1 섹션) | 분산 코드를 single GPU에서 검증 (world_size=1, all_reduce no-op) | val/loss 감소, MLflow run 생성 |
| 3 | `01-notebook-based/02` (1×N 섹션) | 같은 함수로 multi-GPU 확장 — launcher만 num_processes 변경 | GPU util N개 모두 사용, throughput ~N× |
| 4 | `01-notebook-based/03` (M×N) | multi-node 토폴로지 — worker rank-0 attach 패턴 학습 | Spark UI에서 barrier stage 확인, worker GPU metrics |
| 5 | `01-notebook-based/04`, `05` (Lightning 1×1/1×N, M×N) | 같은 모델·데이터로 Lightning Trainer 비교 | MLFlowLogger 자동 동작, EarlyStopping callback |
| 6 | `02-script-based/` 로 이전 | 노트북 셀 코드를 `.py` 모듈로 추출. by-value pickling, sys.path 보강 패턴 학습 | 같은 결과를 동일 클러스터에서 재현 |
| 7 | `03-custom-package-script-based/` 로 이전 | wheel로 패키징 — child 가시성·재현성·subprocess launcher 호환 | `%pip install ./dist/*.whl` 후 Accelerate 노트북 (08) 동작 |

각 단계가 어떤 학습 효과를 주는지 정리하면 다음과 같습니다.

- **01 → 02**: 노트북 셀 안의 학습 코드를 모듈화하면 by-value pickling 한계와 import 경로 함정을 마주하게 됩니다. PoC에서 운영 패턴으로 전환할 때 반드시 거쳐야 하는 단계입니다.
- **02 → 03**: 같은 클러스터라도 child가 다른 컨텍스트(Accelerate subprocess, Job task)로 도는 시나리오에서는 `sys.path` 패턴이 깨집니다. wheel install이 이 문제를 일관되게 해결합니다 ([`00-foundations/env-library-management.md`](00-foundations/env-library-management.md)).

각 행의 학습 시간 ballpark은 행 README의 "기대 결과" 표를 참고하세요.

## 📚 참고

- [`docs/`](docs/README.md) — 외부 자료 인덱스, 공식 노트북 미러, 작업 초기 메모.

## ⚠️ 본 쿡북의 스코프

쿡북이 다루는 범위와 그렇지 않은 범위를 분명히 해 둡니다.

- **다룹니다**: Two-Tower MLP 추천 모델, DDP, TorchDistributor, Lightning, Accelerate, multi-node 데이터 병렬.
- **다루지 않습니다**: FSDP·DeepSpeed(모델 샤딩 불필요), LLM fine-tuning, TorchRec 임베딩 샤딩, vision/multimodal, MosaicML Composer, Ray Train.
- **목표 학습 시간**: 각 노트북 섹션 15분 이내(MovieLens 25M, `max_steps_per_epoch=200`로 epoch당 step 캡).
