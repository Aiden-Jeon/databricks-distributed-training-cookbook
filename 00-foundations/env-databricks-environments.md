# Databricks 실행 환경

본 쿡북은 **Classic GPU 클러스터만** 다룹니다. Multi-node DDP를 지원하는 유일한 옵션이고, 본 쿡북의 매트릭스(1×1 / 1×N / M×N) 전체를 같은 클러스터 종류로 커버할 수 있기 때문입니다.

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
- `nvidia-ml-py`는 MLflow `log_system_metrics=True`가 GPU 메트릭을 수집할 때 필요합니다([debug-common-pitfalls.md §9](debug-common-pitfalls.md)).

다른 DBR 버전을 사용한다면 torch·NCCL·CUDA 조합이 본 쿡북과 달라질 수 있습니다. 그 경우 행 README의 "클러스터 권장 사양" 표를 참고해 인스턴스와 라이브러리를 다시 조정해야 합니다.

## 토폴로지별 클러스터 모양

상세 사양은 [`env-cluster-recipes.md`](env-cluster-recipes.md)에 정리되어 있습니다. 핵심만 요약하면 다음 표와 같습니다.

| 토폴로지 | Single node 토글 | Workers | TorchDistributor `local_mode` |
|----------|-----------------|---------|------------------------------|
| 1×1 | ON | 0 | `True` (또는 직접 호출) |
| 1×N | ON | 0 | `True` |
| M×N | OFF | M | `False` |

driver는 모든 토폴로지에서 GPU 인스턴스로 둡니다. `local_mode=True`에서는 학습용으로 직접 쓰이고, `local_mode=False`에서는 single-node 토글이 OFF여도 driver-side 디버그·검증용으로 활용합니다.

## 참고

자세한 내용은 다음 자료를 참조하세요.

- [Databricks: TorchDistributor 문서](https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor)
- [Databricks: DBR 17.3 LTS ML release notes](https://docs.databricks.com/aws/en/release-notes/runtime/17.3lts-ml)
- 클러스터 세팅 상세: [`env-cluster-recipes.md`](env-cluster-recipes.md)
