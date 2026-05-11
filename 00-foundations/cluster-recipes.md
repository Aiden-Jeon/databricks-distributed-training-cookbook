# Cluster Recipes

토폴로지·데이터 규모에 따른 권장 인스턴스 사양. 본 쿡북은 **AWS만 대상으로 한다** (Azure/GCP는 스코프 밖). 가격은 변하므로 카탈로그 가격 비교는 별도로 확인한다.

기준 모델은 [recommender-baseline.md](recommender-baseline.md)의 Two-Tower MLP. 셀 크기는 임베딩 테이블 규모와 학습 데이터 행 수로 결정된다.

## 1×1 GPU (단일 노드 / 단일 GPU)

| 셀 규모 | 인스턴스 | GPU |
|--------|---------|-----|
| 1×1 (소형, ~10M params) | `g5.2xlarge` | 1× A10G 24GB |

- 임베딩 테이블이 작아 어떤 최신 GPU든 충분.
- 15분 budget 안에서 1 epoch 학습.

## 1×N GPU (단일 노드 / 다중 GPU)

| 셀 규모 | 인스턴스 | GPU 수 |
|--------|---------|-------|
| 1×N (중형, ~200M params) | `g5.12xlarge` | 4× A10G 24GB |
| 1×N (확장) | `p4d.24xlarge` | 8× A100 40GB |

- DDP는 임베딩 테이블을 GPU마다 복제하므로 GPU 메모리에 임베딩 + 배치 + 옵티마이저 상태가 모두 들어가야 한다.

## M×N GPU (다중 노드)

| 셀 규모 | 권장 구성 | 비고 |
|--------|----------|------|
| M×N (대형, ~수 GB params) | 2 노드 × `g5.12xlarge` | TorchDistributor `num_processes = M*N` |
| M×N (확장) | 2~4 노드 × `p4d.24xlarge` | inter-node bandwidth 중요 (EFA 권장) |

> M×N 셀의 학습 패턴 자체는 같은 노드 수에서 데이터 크기를 늘려 throughput을 확장한다. 노드 간 통신 비용이 작은 모델에서는 무시 못 하므로, 본격 운영 전에 노드 수별 throughput을 측정한다.

## 공통 권장

- DBR: ML GPU 15.x 이상.
- 노드 간 네트워크: AWS EFA 가능하면 활성화.
- Autoscaling: 학습 잡에서는 **끈다** (분산 학습은 노드 수 고정 가정).
- Driver: GPU가 있는 인스턴스로 둔다 (TorchDistributor의 rank 0가 driver에서 동작).

## 참고

- AI Runtime 분산 학습 예제 인덱스: https://docs.databricks.com/aws/en/machine-learning/sgc-examples/gpu-distributed-training
