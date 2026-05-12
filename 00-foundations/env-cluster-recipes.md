# Cluster Recipes

토폴로지에 따른 권장 인스턴스 사양. 본 쿡북은 **AWS만 대상으로 합니다** (Azure/GCP는 스코프 밖). 가격은 변하므로 카탈로그 가격 비교는 별도로 확인합니다.

기준 모델은 [concepts-recommender-baseline.md](concepts-recommender-baseline.md)의 Two-Tower MLP, 데이터셋은 MovieLens 25M 고정.

본 쿡북은 ML-25M 단일 데이터셋 + 단일 모델 config로 통일했습니다 ([concepts-recommender-baseline.md](concepts-recommender-baseline.md)). 1×1 / 1×N / M×N의 차이는 **클러스터 모양과 launcher 설정**입니다.

## 1×1 GPU (단일 노드 / 단일 GPU)

| 항목 | 권장 |
|------|------|
| 인스턴스 | `g5.2xlarge` (1× A10G 24GB) 또는 `g5.12xlarge`의 1 GPU만 사용 |

- 임베딩 테이블(~15M params)이 작아 어떤 최신 GPU든 충분.
- 15분 budget 안에서 충분한 epoch 학습.

## 1×N GPU (단일 노드 / 다중 GPU)

| 항목 | 권장 |
|------|------|
| 인스턴스 | `g5.12xlarge` (4× A10G 24GB) |
| 확장 | `p4d.24xlarge` (8× A100 40GB) |

- DDP는 임베딩 테이블을 GPU마다 복제하므로 GPU 메모리에 임베딩 + 배치 + 옵티마이저 상태가 모두 들어가야 합니다 (현재 모델 기준 충분히 여유).

## M×N GPU (다중 노드)

| 항목 | 권장 |
|------|------|
| 기본 | 2 노드 × `g5.12xlarge` (TorchDistributor `num_processes = M*N`) |
| 확장 | 2~4 노드 × `p4d.24xlarge` (inter-node bandwidth 중요, EFA 권장) |

> M×N의 가치는 **데이터 처리량 확장**입니다. 같은 데이터셋·같은 모델에서 batch_size를 GPU 총 수에 비례해 키우고 epoch당 wall-clock을 단축합니다. 노드 간 통신 비용이 작은 모델에서는 무시 못 하므로, 본격 운영 전에 노드 수별 throughput을 측정합니다.

## 공통 권장

- DBR: 17.3 LTS ML (GPU 노드 타입을 선택하면 GPU 이미지가 자동 적용됨).
- 노드 간 네트워크: AWS EFA 가능하면 활성화.
- Autoscaling: 학습 잡에서는 **끈다** (분산 학습은 노드 수 고정 가정).
- Driver: GPU가 있는 인스턴스로 둡니다 (TorchDistributor의 rank 0가 driver에서 동작).

## 참고

- AI Runtime 분산 학습 예제 인덱스: https://docs.databricks.com/aws/en/machine-learning/sgc-examples/gpu-distributed-training
