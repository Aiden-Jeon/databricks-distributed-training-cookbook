# Databricks 실행 환경 비교

Databricks에서 분산 학습을 돌릴 수 있는 환경은 세 가지다. 셀을 선택하기 전 이 표로 환경을 먼저 정한다.

## 🧱 3가지 환경

| 항목 | Classic GPU 클러스터 | Serverless GPU | AI Runtime (SGC) |
|------|---------------------|----------------|-------------------|
| 노드 수 | 1 ~ M (수동 설정) | 1 (단일) | 1 (단일, 워크로드형) |
| GPU 수 | 인스턴스 타입으로 결정 | 인스턴스 타입으로 결정 | 인스턴스 타입으로 결정 |
| Multi-node | ✅ TorchDistributor | ❌ | ❌ |
| 런타임 관리 | DBR ML GPU 이미지 직접 선택 | Databricks 관리 | Databricks 관리, 학습용 패키지 사전 설치 |
| 시작 시간 | 분 단위 | 초~분 단위 | 초~분 단위 |
| 권장 시나리오 | M×N GPU, multi-node DDP | 1×1, 1×N PoC | `@distributed` 사용 노트북, 1×N |
| 본 쿡북에서 사용 | 01-3, 02-3, 03-3, 04-3 (multi-node) | 01-1, 02-1, 03-1, 04-1 | 01-2, 02-2, 03-2, 04-2 |

## ⚠️ 환경 선택 규칙

1. **Multi-node가 필요한가?** → 무조건 Classic GPU 클러스터.
2. **`@distributed` 데코레이터를 쓰는가?** → AI Runtime이 가장 매끄럽다.
3. **단순 PoC인가?** → Serverless GPU가 가장 빠르게 시작한다.

## DBR 버전

본 쿡북은 다음을 가정한다:

- Classic: DBR 15.x ML GPU 이상 (CUDA 12, PyTorch 2.3+)
- AI Runtime: 최신 SGC(Serverless GPU Compute) 이미지
- Serverless GPU: 워크스페이스 설정에 따라 자동

각 셀 README의 "클러스터 권장 사양" 표에서 다시 명시한다.

## 참고

- Multi-GPU workload (`@distributed`): https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training
- TorchDistributor: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor
