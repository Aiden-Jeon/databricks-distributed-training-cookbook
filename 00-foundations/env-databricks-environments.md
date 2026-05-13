# Databricks 실행 환경

본 쿡북은 **Classic GPU 클러스터만** 다룹니다. Multi-node DDP를 지원하는 유일한 옵션이고, 본 쿡북의 매트릭스(1×1 / 1×N / M×N) 전체를 같은 클러스터 종류로 커버할 수 있기 때문입니다. 또한 인스턴스 권장은 **AWS만 대상**으로 합니다(Azure·GCP는 스코프 밖). 가격은 수시로 바뀌므로 카탈로그 가격 비교는 별도로 확인하시기 바랍니다.

> Databricks에는 Serverless GPU(= AI Runtime / SGC)라는 또 다른 GPU 컴퓨트가 있지만 single-node 전용이고 `@distributed` 데코레이터처럼 별도 인터페이스를 갖습니다. 본 쿡북에서는 다루지 않습니다.

## Classic GPU 클러스터

Classic GPU 클러스터의 주요 특성은 다음과 같습니다.

| 항목 | 값 |
|------|-----|
| 노드 수 | 1 ~ M (수동 설정, Autoscaling은 분산 학습에서 OFF) |
| GPU 수 | 인스턴스 타입으로 결정 (`g5.2xlarge` = 1, `g5.12xlarge` = 4, `p4d.24xlarge` = 8) |
| Multi-node | TorchDistributor (`local_mode=False`)로 지원 |
| 런타임 | DBR ML GPU 이미지 직접 선택 |
| 시작 시간 | 분 단위 (cold start ~3-5분) |
| 본 쿡북에서 사용 | 모든 노트북 (1×1 / 1×N / M×N) |

## DBR 버전

본 쿡북은 **DBR 17.3 LTS ML**을 가정합니다. 사전 설치된 핵심 버전은 다음과 같습니다.

| 항목 | 버전 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.7.0 |
| CUDA | 12.6 |
| cuDNN | 9.5.1 |
| NCCL | 2.26.2 |
| MLflow | 3.0.1 |
| accelerate | 1.5.2 |
| transformers | 4.51.3 |

모든 행의 `00-setup.ipynb`에서 추가로 설치하는 패키지는 다음과 같습니다.

- `lightning==2.5.1`은 DBR에 포함되어 있지 않아 명시적으로 설치합니다.
- `nvidia-ml-py`는 MLflow `log_system_metrics=True`가 GPU 메트릭을 수집할 때 필요합니다([debug-common-pitfalls.md §9](../docs/debug-common-pitfalls.md)).

다른 DBR 버전을 사용한다면 torch·NCCL·CUDA 조합이 본 쿡북과 달라질 수 있습니다. 그 경우 아래 인스턴스 권장도 함께 다시 조정해야 합니다.

## 토폴로지별 클러스터 모양

기준 모델은 [concepts-recommender-baseline.md](concepts-recommender-baseline.md)의 Two-Tower MLP이고, 데이터셋은 MovieLens 25M으로 고정되어 있습니다. 본 쿡북은 ML-25M 단일 데이터셋과 단일 모델 config로 통일했기 때문에 1×1 / 1×N / M×N의 차이는 **클러스터 모양과 launcher 설정**에서만 나타납니다.

토폴로지별 클러스터 토글 요약은 다음과 같습니다.

| 토폴로지 | Single node 토글 | Workers | TorchDistributor `local_mode` |
|----------|-----------------|---------|------------------------------|
| 1×1 | ON | 0 | `True` (또는 직접 호출) |
| 1×N | ON | 0 | `True` |
| M×N | OFF | M | `False` |

driver는 모든 토폴로지에서 GPU 인스턴스로 둡니다. `local_mode=True`에서는 학습용으로 직접 쓰이고, `local_mode=False`에서는 single-node 토글이 OFF여도 driver-side 디버그·검증용으로 활용합니다.

### 1×1 GPU (단일 노드 / 단일 GPU)

가장 단순한 단일 GPU 구성입니다.

| 항목 | 권장 |
|------|------|
| 인스턴스 | `g5.2xlarge` (1× A10G 24GB) 또는 `g5.12xlarge`의 1 GPU만 사용 |

임베딩 테이블이 ~15M params 수준으로 작기 때문에 최신 GPU 중 무엇을 골라도 충분히 들어갑니다. 15분 budget 안에서 충분한 epoch을 돌릴 수 있습니다.

### 1×N GPU (단일 노드 / 다중 GPU)

같은 노드 안에서 GPU를 여러 장 쓰는 구성입니다.

| 항목 | 권장 |
|------|------|
| 인스턴스 | `g5.12xlarge` (4× A10G 24GB) |
| 확장 | `p4d.24xlarge` (8× A100 40GB) |

DDP는 임베딩 테이블을 GPU마다 복제하므로 GPU 메모리에 임베딩과 배치, 옵티마이저 상태가 모두 들어가야 합니다. 현재 모델 크기에서는 충분히 여유가 있습니다.

### M×N GPU (다중 노드)

여러 노드를 묶어 GPU를 총 M×N개 사용하는 구성입니다.

| 항목 | 권장 |
|------|------|
| 기본 | 2 노드 × `g5.12xlarge` (TorchDistributor `num_processes = M*N`) |
| 확장 | 2~4 노드 × `p4d.24xlarge` (inter-node bandwidth 중요, EFA 권장) |

> M×N의 가치는 **데이터 처리량 확장**에 있습니다. 같은 데이터셋과 같은 모델에서 batch_size를 GPU 총 수에 비례해 키우고 epoch당 wall-clock을 단축합니다. 노드 간 통신 비용은 작은 모델에서 무시할 수 없는 수준이므로, 본격 운영 전에 노드 수별 throughput을 측정하는 것이 좋습니다.

## 공통 권장

토폴로지와 무관하게 적용하는 공통 설정입니다.

- DBR은 17.3 LTS ML을 사용합니다(GPU 노드 타입을 선택하면 GPU 이미지가 자동으로 적용됩니다).
- 노드 간 네트워크는 가능하면 AWS EFA를 활성화합니다.
- 학습 잡에서는 Autoscaling을 **끕니다**(분산 학습은 노드 수 고정을 가정합니다).
- Driver는 GPU가 있는 인스턴스로 둡니다(TorchDistributor의 rank 0가 driver에서 동작합니다).

## 참고

자세한 내용은 다음 자료를 참조하세요.

- [Databricks: TorchDistributor 문서](https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor)
- [Databricks: DBR 17.3 LTS ML release notes](https://docs.databricks.com/aws/en/release-notes/runtime/17.3lts-ml)
- [Databricks AI Runtime: 분산 학습 예제 인덱스](https://docs.databricks.com/aws/en/machine-learning/sgc-examples/gpu-distributed-training)
