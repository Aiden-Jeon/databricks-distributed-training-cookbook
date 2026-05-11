# Databricks Distributed Training Cookbook

Databricks 환경에서 **추천 모델(Two-Tower MLP)**을 분산학습하는 패턴을 **코드 조직 방식 × 학습 launcher** 매트릭스로 정리한 쿡북.

## 🎯 누구를 위한 쿡북인가

- PyTorch 추천 모델을 처음 Databricks에서 분산학습하려는 ML 엔지니어
- 노트북 PoC → 스크립트 분리 → 패키지화 경로를 한 곳에서 보고 싶은 팀
- 같은 모델을 TorchDistributor / Lightning / Accelerate 중 어느 launcher로 돌릴지 비교하려는 팀

## 🧭 매트릭스

|                  | TorchDistributor | PyTorch Lightning | HF Accelerate |
|------------------|------------------|-------------------|---------------|
| **01-notebook-based** | [`02-train_torch_distributor_single_node.ipynb`](01-notebook-based/02-train_torch_distributor_single_node.ipynb) (1×1, 1×N)<br>[`03-train_torch_distributor_multi_node.ipynb`](01-notebook-based/03-train_torch_distributor_multi_node.ipynb) (M×N) | [`04-train_pytorch_lightning_single_node.ipynb`](01-notebook-based/04-train_pytorch_lightning_single_node.ipynb) (1×1, 1×N)<br>[`05-train_pytorch_lightning_multi_node.ipynb`](01-notebook-based/05-train_pytorch_lightning_multi_node.ipynb) (M×N) | — (CLI 기반이라 N/A) |
| **02-script-based** | 02 (1×1) · 03 (1×N) · 04 (M×N) | 05 (1×1) · 06 (1×N) · 07 (M×N) | 08 (1×1) · 09 (1×N) · 10 (M×N) |
| **03-custom-package-script-based** | [`02-launch_torch_distributor.ipynb`](03-custom-package-script-based/02-launch_torch_distributor.ipynb) | [`03-launch_lightning_trainer.ipynb`](03-custom-package-script-based/03-launch_lightning_trainer.ipynb) | [`04-launch_accelerator.ipynb`](03-custom-package-script-based/04-launch_accelerator.ipynb) |

02-script-based는 launcher × topology별로 노트북을 분리했습니다 (한 노트북에서 `TorchDistributor.run`을 연속 호출 시 py4j callback이 단절되는 현상 회피). 노트북 목록은 [`02-script-based/README.md`](02-script-based/README.md). 03 행은 TBD.

### 행: 코드 조직 방식

| 행 | 차별점 |
|----|--------|
| 01-notebook-based | 학습 코드를 노트북 셀 안에 작성. 가장 빠른 PoC 경로. |
| 02-script-based | 같은 디렉터리에 `.py` 모듈(model/trainer)을 두고 노트북은 driver 역할. 상대 경로 import. |
| 03-custom-package-script-based | 학습 코드를 `pyproject.toml` 기반 설치 가능 패키지로 분리. `uv build` → wheel install 후 패키지명으로 import. 여러 노트북/job 재사용, CI/CD에 적합. |

### 열: 학습 launcher

| 열 | API |
|----|-----|
| TorchDistributor | `TorchDistributor(num_processes, local_mode, use_gpu).run(fn, ...)` |
| PyTorch Lightning | `Trainer(accelerator="gpu", devices=N, num_nodes=M, strategy="ddp").fit(...)` |
| HF Accelerate | `%sh accelerate launch --config_file <cfg>.yaml train.py` |

### 토폴로지 (노트북 내부 섹션)

| 섹션 | 의미 | 권장 환경 |
|------|------|----------|
| 1×1 GPU | 1 노드 / 1 GPU | Classic GPU 단일 인스턴스 또는 Serverless GPU |
| 1×N GPU | 1 노드 / N GPU | Classic 다중 GPU 인스턴스 (g5.12xlarge 등) |
| M×N GPU | M 노드 / 노드당 N GPU | Classic 멀티 노드 (TorchDistributor 백엔드) |

## 🧱 공통 스택

- 모델: Two-Tower MLP 추천 모델 ([00-foundations/recommender-baseline.md](00-foundations/recommender-baseline.md))
- 학습 라이브러리: `torch`, `accelerate`, `pytorch-lightning`
- 병렬화: PyTorch **DDP** (FSDP/DeepSpeed는 다루지 않음)
- 데이터: MovieLens 25M (implicit feedback 변환) → UC Volume parquet shard → torch DataLoader
- 체크포인트: Unity Catalog Volumes
- 실험 추적: MLflow (3.0+ 패턴)
- Multi-node launcher: TorchDistributor

## 🗺️ 시작 가이드

1. 처음이면 [`00-foundations/`](00-foundations/)부터 읽습니다.
   - 기준 모델 정의, DDP 개념, Classic vs Serverless GPU vs AI Runtime, 클러스터 권장 사양, 데이터 로딩, MLflow, UC Volumes, 흔한 함정.
2. 매트릭스에서 자신의 시나리오에 가장 가까운 셀(노트북)을 엽니다.
3. 해당 행(`01-notebook-based` / `02-script-based` / `03-custom-package-script-based`)의 README "실행 순서"를 따릅니다.
4. 토폴로지 섹션을 1×1 → 1×N → M×N 순으로 확장합니다.

## 📚 참고

- [`99-references/`](99-references/) — 공식 노트북 미러, 외부 자료 인덱스, 이전 프로젝트 snippets

## ⚠️ 본 쿡북의 스코프

- **다룹니다**: Two-Tower MLP 추천 모델, DDP, TorchDistributor, Lightning, Accelerate, multi-node 데이터 병렬.
- **다루지 않습니다**: FSDP·DeepSpeed (모델 샤딩 불필요), LLM fine-tuning, TorchRec 임베딩 샤딩, vision/multimodal, MosaicML Composer, Ray Train.
- **목표 학습 시간**: 각 노트북 섹션 ≤ 15분 (MovieLens 25M, `max_steps_per_epoch=200`로 epoch당 step 캡).
