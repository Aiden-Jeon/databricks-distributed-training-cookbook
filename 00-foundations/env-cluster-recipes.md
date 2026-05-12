# Cluster Recipes

이 문서는 토폴로지에 따른 권장 인스턴스 사양을 정리합니다. 본 쿡북은 **AWS만 대상으로 합니다**(Azure와 GCP는 스코프 밖입니다). 가격은 수시로 바뀌므로 카탈로그 가격 비교는 별도로 확인하시기 바랍니다.

기준 모델은 [concepts-recommender-baseline.md](concepts-recommender-baseline.md)의 Two-Tower MLP이고, 데이터셋은 MovieLens 25M으로 고정되어 있습니다. 본 쿡북은 ML-25M 단일 데이터셋과 단일 모델 config로 통일했기 때문에 1×1 / 1×N / M×N의 차이는 **클러스터 모양과 launcher 설정**에서만 나타납니다.

## 1×1 GPU (단일 노드 / 단일 GPU)

가장 단순한 단일 GPU 구성입니다.

| 항목 | 권장 |
|------|------|
| 인스턴스 | `g5.2xlarge` (1× A10G 24GB) 또는 `g5.12xlarge`의 1 GPU만 사용 |

임베딩 테이블이 ~15M params 수준으로 작기 때문에 최신 GPU 중 무엇을 골라도 충분히 들어갑니다. 15분 budget 안에서 충분한 epoch을 돌릴 수 있습니다.

## 1×N GPU (단일 노드 / 다중 GPU)

같은 노드 안에서 GPU를 여러 장 쓰는 구성입니다.

| 항목 | 권장 |
|------|------|
| 인스턴스 | `g5.12xlarge` (4× A10G 24GB) |
| 확장 | `p4d.24xlarge` (8× A100 40GB) |

DDP는 임베딩 테이블을 GPU마다 복제하므로 GPU 메모리에 임베딩과 배치, 옵티마이저 상태가 모두 들어가야 합니다. 현재 모델 크기에서는 충분히 여유가 있습니다.

## M×N GPU (다중 노드)

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

- [Databricks AI Runtime: 분산 학습 예제 인덱스](https://docs.databricks.com/aws/en/machine-learning/sgc-examples/gpu-distributed-training)
