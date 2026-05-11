# Databricks Distributed Training Cookbook

Databricks 환경에서 분산 학습(distributed training)을 수행하는 레시피와 예제를 모은 쿡북.

## 행동 가이드라인

속도보다 정확성을 우선한다. 사소한 작업에서는 계획·설명의 길이만 줄이되, 아래 원칙 자체는 유지한다.

### 1. 먼저 생각하기

가정하지 않는다. 혼란을 숨기지 않는다. 트레이드오프를 드러낸다.

- 관련 코드, 테스트, 설정을 먼저 읽고 기존 패턴을 파악한 뒤 변경한다.
- 작업 전 가정을 명시한다. 불확실하면 질문한다.
- 해석이 여러 가지 가능하면 제시하고 선택을 구한다. 임의로 고르지 않는다.
- 더 단순한 방법이 있으면 말한다. 필요하면 반론을 제기한다.
- 불명확한 점이 있으면 멈추고, 무엇이 불명확한지 명시하고, 질문한다.

### 2. 단순성 우선

문제를 해결하는 최소한의 코드만 작성한다. 추측성 확장은 하지 않는다.

- 요청된 범위 밖의 기능을 추가하지 않는다.
- 한 번만 쓰이는 코드에 과도한 추상화를 만들지 않는다.
- 요청되지 않은 "유연성"이나 "확장성"을 위한 구조를 만들지 않는다.
- 실제 호출 경로에서 관측 가능한 실패만 다룬다. 이론적으로만 가능한 시나리오에 방어 코드를 추가하지 않는다.
- 더 짧고 더 읽기 쉬운 방법이 있으면 단순화한다.

기준: "시니어 엔지니어가 보고 과도하다고 할 것인가?" 그렇다면 단순화한다.

### 3. 최소 변경

건드려야 할 것만 건드린다. 정리는 자기가 만든 혼란에 대해서만 한다.

- 요청을 만족시키기 위해 필요한 범위에서만 수정한다. 인접한 코드, 주석, 포매팅을 임의로 "개선"하지 않는다.
- 문제가 없는 코드를 리팩터링하지 않는다. 단, 변경의 정합성을 위해 필요하면(타입 수정, 테스트 갱신 등) 수반 변경은 한다.
- 기존 스타일에 맞춘다. 본인이 다르게 하고 싶어도.
- 관련 없는 문제를 발견하면 언급만 하고 수정하지 않는다.
- 본인의 변경으로 발생한 미사용 임포트, 변수, 함수는 정리한다.

기준: 변경된 모든 줄이 사용자의 요청으로 직접 추적 가능해야 한다.

### 4. 목표 중심 실행

성공 기준을 정의하고, 검증될 때까지 반복한다.

작업을 검증 가능한 목표로 변환한다:
- "유효성 검사 추가" → "잘못된 입력에 대한 테스트 작성 → 테스트 통과시키기"
- "버그 수정" → "재현 테스트 작성 → 수정 → 테스트 통과 확인"
- "리팩터링" → "기존 테스트 통과 확인 → 변경 → 테스트 재확인"

복수 단계 작업은 간략한 계획을 먼저 서술한다:
```
1. [단계] → 확인: [검증 방법]
2. [단계] → 확인: [검증 방법]
3. [단계] → 확인: [검증 방법]
```

명확한 성공 기준이 있으면 자율적으로 진행한다. 모호한 기준("알아서 해줘")이면 먼저 명확화한다.

검증이 불가능한 환경(테스트 미구축, 외부 의존성 등)에서는 가능한 가장 강한 검증을 수행하되, 미검증 범위와 잔여 리스크를 명시한다.

---

## 프로젝트 개요

Databricks에서 **추천 모델(Two-Tower MLP)**을 분산학습하는 패턴을 **코드 조직 방식 × 학습 launcher** 매트릭스로 정리한 쿡북.

- 코드 조직 방식 (행): 01-notebook-based / 02-script-based / 03-custom-package-script-based
- 학습 launcher (열): TorchDistributor / PyTorch Lightning / HF Accelerate (Accelerate는 01-notebook-based에서는 다루지 않음)
- 컴퓨트 토폴로지는 **각 학습 노트북 내부의 섹션**으로 비교: 1×1 GPU / 1×N GPU / M×N GPU
- Multi-node 백엔드: TorchDistributor (Classic GPU 클러스터)
- 기준 모델은 `00-foundations/recommender-baseline.md` 한 곳에 정의. **데이터셋·모델 config 모두 단일**(ML-25M, 단일 CONFIG)이고, 토폴로지 차이는 launcher 설정·batch_size로만 표현.

## 기술 스택

| 영역 | 사용 |
|------|------|
| 언어 | Python 3.10+ |
| ML 프레임워크 | PyTorch |
| 분산 학습 | PyTorch **DDP**, TorchDistributor, HF Accelerate, PyTorch Lightning |
| 실행 환경 | Databricks (Classic GPU 클러스터, Serverless GPU, AI Runtime) |
| 데이터 | MovieLens 25M (implicit feedback 변환) → Unity Catalog Volumes parquet shard |
| 실험 추적 | MLflow 3.0+ (system metrics, per-epoch LoggedModel, dataset linking) |
| 노트북 포맷 | `.ipynb` (Jupyter) — Databricks Repos에서 그대로 열림 |
| 패키징 | `uv` + `hatchling` (03-custom-package-script-based의 wheel 빌드) |

**사용 안 함**: FSDP (모델 샤딩 불필요), DeepSpeed, TorchRec, HuggingFace Transformers/TRL/PEFT (LLM fine-tuning 스택은 이번 쿡북 스코프 밖), MosaicML Composer, Ray Train.

## 디렉토리 구조

```
.
├── README.md                                       # 쿡북 메인, 매트릭스 + 네비게이션
├── 00-foundations/                                 # 공통 개념 / 기준 모델 / 환경 / 데이터 / 추적 / 함정
├── 01-notebook-based/                              # 행 1: 노트북 only
│   ├── 00-setup.ipynb
│   ├── 01-data_prep.ipynb
│   ├── 02-train_torch_distributor.ipynb            # 섹션 1×1 / 1×N / M×N
│   └── 03-train_pytorch_lightning.ipynb            # 섹션 1×1 / 1×N / M×N
├── 02-script-based/                                # 행 2: .py 모듈 분리
│   ├── model.py, torch_distributor_trainer.py, lightning_trainer.py
│   ├── 00-setup.ipynb, 01-data_prep.ipynb
│   ├── 02-launch_torch_distributor.ipynb
│   ├── 03-launch_lightning_trainer.ipynb
│   └── 04-launch_accelerator.ipynb
├── 03-custom-package-script-based/                 # 행 3: 설치 가능 패키지
│   ├── custom_packages/                            # uv build → wheel install
│   │   ├── pyproject.toml
│   │   └── src/recommender_pkg/
│   ├── 00-setup.ipynb, 01-data_prep.ipynb
│   ├── 02-launch_torch_distributor.ipynb
│   ├── 03-launch_lightning_trainer.ipynb
│   └── 04-launch_accelerator.ipynb
├── 99-references/                                  # 공식 노트북 / 외부 자료 / 이전 프로젝트 snippets
└── docs/                                           # 부가 문서
```

## 개발 명령어

이 레포는 로컬 빌드 산출물이 없는 **레퍼런스 코드 모음**이다. 실행은 모두 Databricks 워크스페이스에서 일어난다. 예외: 03-custom-package-script-based는 wheel 빌드만 로컬 또는 노트북에서 `uv build` 한 번 실행.

| launcher | 1×1 GPU | 1×N GPU | M×N GPU |
|----------|---------|---------|---------|
| TorchDistributor | `TorchDistributor(num_processes=1, local_mode=True).run(fn, ...)` | `TorchDistributor(num_processes=N, local_mode=True).run(fn, ...)` | `TorchDistributor(num_processes=M*N, local_mode=False).run(fn, ...)` |
| Lightning | `Trainer(devices=1, num_nodes=1)` | `Trainer(devices=N, num_nodes=1, strategy="ddp")` | `Trainer(devices=N, num_nodes=M, strategy="ddp")` |
| Accelerate | `accelerate launch --num_processes 1 ...` | `accelerate launch --num_processes N ...` | `accelerate launch --multi_gpu --num_machines M --num_processes M*N ...` |

## 코딩 컨벤션

- 한국어로 작성한다.
- 이모지는 섹션 헤더에만 쓴다. 본문에서는 쓰지 않는다.
- 비교/매핑이 등장하면 표(table)를 적극 사용한다.
- 각 행 디렉터리의 README는 100~200줄을 목표로 한다. 너무 길어지면 분할한다.
- 노트북은 `.ipynb` 포맷. 토폴로지(1×1 / 1×N / M×N) 섹션은 학습 노트북 안에서 markdown `##` 헤더로 구분.
- "Generated by Claude" 같은 메타 코멘트는 작성하지 않는다.

## 테스트 규칙

- 이 쿡북에는 단위 테스트가 없다.
- 검증 기준은 **각 노트북 섹션이 명시된 Databricks 클러스터 사양에서 끝까지 실행됨**이다.
- 변경 시 최소한 영향을 받는 행 README의 "실행 순서"를 따라가 재현 가능성을 확인한다.

## Git 규칙

- 커밋은 행 단위 또는 노트북 단위로 작게 끊는다 (e.g. `feat(01-notebook-based): TorchDistributor 1xN 섹션`).
- 메인 브랜치는 `main`. 큰 변경은 별도 브랜치 + PR.

## 운영 제약

- **FSDP·DeepSpeed 관련 내용은 어디에도 추가하지 않는다** (이 쿡북의 명시적 비대상). 모델은 단일 GPU에 들어가는 크기를 유지한다.
- Multi-node는 TorchDistributor 백엔드만 다룬다. 다른 launcher(MPI, deepspeed launcher 등)는 다루지 않는다.
- 모델은 `00-foundations/recommender-baseline.md`의 Two-Tower MLP를 기준으로 한다. 다른 추천 모델(SASRec, BERT4Rec 등)이나 LLM으로 확장하지 않는다.
- 데이터셋은 MovieLens 25M 고정. small/medium/large 같은 scale 매트릭스를 다시 도입하지 않는다.
- 각 노트북 섹션의 학습 시간은 **15분 이내**를 목표로 한다. epoch 상한 + `max_steps_per_epoch`로 budget을 컨트롤한다.
- 03-custom-package-script-based의 `model.py`, `*_trainer.py` 시그니처는 02-script-based의 동일 이름 파일과 동작·시그니처가 같아야 한다. 차이는 import 경로뿐.
