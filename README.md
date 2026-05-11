# Databricks Distributed Training Cookbook

Databricks 환경에서 HuggingFace NLP 모델(Llama, GPT-OSS 계열)을 분산학습하는 패턴을 **실행 방식 × 컴퓨트 토폴로지** 매트릭스로 정리한 쿡북.

## 🎯 누구를 위한 쿡북인가

- HuggingFace 모델을 처음 Databricks에서 분산학습하려는 ML 엔지니어
- 노트북 PoC → 스크립트 기반 운영 → multi-node 스케일아웃 경로를 한 곳에서 보고 싶은 팀
- 같은 모델을 DDP / FSDP / Accelerate / Lightning 중 어떤 방식으로 돌릴지 비교하려는 팀

## 🧭 매트릭스

|                | 1×1 GPU | 1×N GPU | M×N GPU |
|----------------|---------|---------|---------|
| **노트북 only**   | [01-1](01-notebook-only/01-single-node-single-gpu/) | [01-2](01-notebook-only/02-single-node-multi-gpu/) | [01-3](01-notebook-only/03-multi-node-multi-gpu/) |
| **스크립트**       | [02-1](02-script-based/01-single-node-single-gpu/)  | [02-2](02-script-based/02-single-node-multi-gpu/)  | [02-3](02-script-based/03-multi-node-multi-gpu/)  |
| **Accelerate**  | [03-1](03-cli-accelerate/01-single-node-single-gpu/) | [03-2](03-cli-accelerate/02-single-node-multi-gpu/) | [03-3](03-cli-accelerate/03-multi-node-multi-gpu/) |
| **Lightning**   | [04-1](04-lightning/01-single-node-single-gpu/)    | [04-2](04-lightning/02-single-node-multi-gpu/)    | [04-3](04-lightning/03-multi-node-multi-gpu/)    |

### 행: 실행 방식

| 행 | 차별점 |
|----|--------|
| 노트북 only | 노트북 셀 안에서 학습 함수를 직접 작성·실행. `@distributed` 데코레이터 또는 `TorchDistributor.run(fn, ...)`. |
| 스크립트 | `train.py`를 분리하고 노트북은 driver 역할만. `TorchDistributor.run("train.py", ...)`. |
| Accelerate | HF Accelerate CLI로 launch. `%sh accelerate launch --config_file <cfg>.yaml train.py`. |
| Lightning | PyTorch Lightning `Trainer(devices, num_nodes, strategy)` API로 분산화. |

### 열: 컴퓨트 토폴로지

| 열 | 의미 | 권장 환경 |
|----|------|----------|
| 1×1 GPU | 1 노드 / 1 GPU | Classic GPU 단일 인스턴스 또는 Serverless GPU |
| 1×N GPU | 1 노드 / N GPU | Classic 다중 GPU 인스턴스 (g5.12xlarge 등) |
| M×N GPU | M 노드 / 노드당 N GPU | Classic 멀티 노드 + TorchDistributor |

## 🧱 공통 스택

- 모델: HuggingFace NLP (Llama-3.2, GPT-OSS 등)
- 학습 라이브러리: `transformers`, `trl`, `accelerate`, `peft`, `pytorch-lightning`
- 병렬화: PyTorch DDP / FSDP (DeepSpeed는 다루지 않음)
- 데이터: Delta Lake → HF Dataset, 대용량은 Mosaic Streaming(MDS)
- 체크포인트: Unity Catalog Volumes
- 실험 추적: MLflow autolog
- Multi-node launcher: TorchDistributor

## 🗺️ 시작 가이드

1. 처음이면 [`00-foundations/`](00-foundations/)부터 읽는다.
   - DDP vs FSDP 차이, Classic vs Serverless GPU vs AI Runtime, 클러스터 권장 사양, 데이터 로딩, MLflow, UC Volumes, 흔한 함정.
2. 위 매트릭스에서 자신의 시나리오에 가장 가까운 셀을 연다.
3. 셀 README의 "🚀 실행 순서"를 따라간다.
4. 한 셀이 동작하면 README 하단의 "➡️ 다음 셀" 링크를 따라 옆/아래로 확장한다.

## 📚 참고

- [`99-references/`](99-references/) — 공식 노트북 미러 및 외부 자료 인덱스.

## ⚠️ 본 쿡북의 스코프

- **다룬다**: HuggingFace NLP fine-tuning, DDP, FSDP, LoRA, TorchDistributor, Accelerate, Lightning.
- **다루지 않는다**: DeepSpeed, pretraining-from-scratch, vision/multimodal, RLHF 전체 파이프라인.
