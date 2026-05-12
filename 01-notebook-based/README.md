# 01 · Notebook-based

> 모든 학습 코드가 노트북 셀 안에 있습니다. 가장 빠른 PoC 경로.

## 🧭 노트북 흐름

| 순서 | 파일 | 역할 | 클러스터 |
|------|------|------|----------|
| 00 | [`00-setup.ipynb`](00-setup.ipynb) | 패키지 설치, UC 경로, 단일 `CONFIG`, MLflow experiment, `DB_HOST/DB_TOKEN` | 아무 클러스터(코드 실행만) |
| 01 | [`01-data_prep.ipynb`](01-data_prep.ipynb) | MovieLens 25M 다운로드 → implicit feedback 변환 → UC Volume parquet shard | 단일 노드 CPU도 가능 |
| 02 | [`02-train_torch_distributor_single_node.ipynb`](02-train_torch_distributor_single_node.ipynb) | TorchDistributor 1×1 / 1×N 섹션 | Single node GPU |
| 03 | [`03-train_torch_distributor_multi_node.ipynb`](03-train_torch_distributor_multi_node.ipynb) | TorchDistributor M×N | Multi node GPU |
| 04 | [`04-train_pytorch_lightning_single_node.ipynb`](04-train_pytorch_lightning_single_node.ipynb) | Lightning 1×1 / 1×N 섹션 | Single node GPU |
| 05 | [`05-train_pytorch_lightning_multi_node.ipynb`](05-train_pytorch_lightning_multi_node.ipynb) | Lightning M×N | Multi node GPU |

Single-node 노트북(02, 04)은 같은 single-node 클러스터에서 둘 다 실행 가능. Multi-node 노트북(03, 05)은 multi-node 클러스터에서 실행. 둘은 클러스터를 분리해서 띄우는 것을 권장 (autoscaling·single-node 토글 설정이 다름).

## 🔀 매트릭스

| | TorchDistributor | Lightning |
|----|---|---|
| **1×1 GPU** | 02 (`TorchDistributor(num_processes=1, local_mode=True).run(fn)` + DDP) | 04 (`Trainer(devices=1)`) |
| **1×N GPU** | 02 (`TorchDistributor(num_processes=N, local_mode=True).run(fn)` + DDP) | 04 (`TorchDistributor(local_mode=True).run(fn)` + `Trainer(strategy='ddp')`) |
| **M×N GPU** | 03 (`TorchDistributor(num_processes=M*N, local_mode=False).run(fn)` + DDP) | 05 (TorchDistributor + `Trainer(devices=N, num_nodes=M, strategy='ddp')`) |

## 🖥️ 클러스터 세팅

같은 학습 launcher라도 single-node / multi-node 노트북은 **별도 클러스터**에서 띄우는 것을 권장합니다 (autoscaling·single-node 토글 설정이 다름). 상세 인스턴스 비교는 [`00-foundations/cluster-recipes.md`](../00-foundations/cluster-recipes.md).

| 항목 | Single-node (02, 04) | Multi-node (03, 05) |
|------|----------------------|----------------------|
| Cluster mode | Single user | Single user |
| Access mode | Dedicated | Dedicated |
| Databricks Runtime | 17.3 LTS ML (CUDA 12.6) | 17.3 LTS ML |
| Single node 토글 | **ON** | **OFF** |
| Driver type | `g5.12xlarge` (4× A10G) | `g5.12xlarge` (4× A10G) |
| Worker type | — | `g5.12xlarge` (4× A10G) |
| Workers | 0 | M (예: 1~3) |
| Autoscaling | off | **off (필수)** |
| 사용 노트북 | 02, 04 (1×1 / 1×N 모두 같은 클러스터) | 03, 05 (M×N) |

만들 때 주의

- **Single node 토글**: 02·04는 ON, 03·05는 OFF. 같은 클러스터를 토글만 바꿔 재사용하면 안 됨 (재생성 필요).
- **Autoscaling**: 03·05는 반드시 OFF. DDP는 학습 시작 시 노드 수가 고정되어야 합니다.
- **Driver도 GPU**: M×N에서 TorchDistributor 코디네이션은 driver에서 일어납니다. CPU driver로 두면 일부 PyTorch import가 실패.
- **DBR 17.3 LTS ML**: 일반 DBR이 아닌 "Machine Learning" 런타임을 선택해야 `torch`, CUDA가 사전 설치. GPU 노드 타입(`g5.*` 등)을 선택하면 GPU 이미지가 자동 적용됩니다.
- 02의 1×1 섹션은 GPU 1개만 사용해도 되므로 `g5.2xlarge` (1× A10G) 클러스터에서 따로 돌려도 됩니다. 1×N과 클러스터를 공유하려면 그냥 `g5.12xlarge`.

H100/A100을 쓰려면 `p4d.24xlarge` 또는 H100 인스턴스로 교체 ([`cluster-recipes.md`](../00-foundations/cluster-recipes.md)).

## 📊 MLflow 로깅

- 모든 토폴로지(1×1 / 1×N / M×N)의 run은 `00-setup`에서 만든 단일 experiment(`/Users/<email>/recommender-notebook-based`)에 누적됩니다. 같은 experiment 안에서 run을 비교하세요.
- **System metrics**(GPU 활용도, 메모리, I/O)는 `mlflow.start_run(..., log_system_metrics=True)`로 수집됩니다.
  - Single-node(02, 04): driver-side에서만 켜도 충분 (driver와 worker가 같은 머신).
  - **Multi-node(03, 05)**: driver는 학습에 참여하지 않으므로 driver-side만 켜면 idle 메트릭만 잡힙니다. 03·05의 학습 함수는 **rank-0 worker에서 `mlflow.start_run(run_id=..., log_system_metrics=True)`로 attach**해 worker 노드의 GPU 메트릭을 함께 기록합니다 ([`00-foundations/mlflow-tracking.md §1`](../00-foundations/mlflow-tracking.md)).

## 📈 기대 결과

ML-25M 데이터(positive+negative ≈ 25M 행) + `max_steps_per_epoch=200` 기준. 환경(NCCL 버전, EFA 유무, 디스크 캐시 상태)에 따라 ±30%는 흔합니다.

| 노트북 | 토폴로지 | 학습 시간 | val/loss (10 epoch 또는 early stop) | GPU util (rank-0 기준) |
|--------|---------|----------|-----------------------------------|----------------------|
| 02 (1×1) | 1 노드 × 1 GPU (A10G) | 3~6분 | ≈ 0.45 ~ 0.55 | 60~85% |
| 02 (1×N) | 1 노드 × 4 GPU (g5.12xlarge) | 2~4분 | ≈ 0.45 ~ 0.55 | 50~80% (DataLoader 병목 시 낮음) |
| 03 (M×N) | 2 노드 × 4 GPU | 2~3분 | ≈ 0.45 ~ 0.55 | 40~70% (EFA 없으면 낮음) |
| 04 (1×1) | Lightning, 1 노드 × 1 GPU | 3~6분 | ≈ 0.45 ~ 0.55 | 60~85% |
| 04 (1×N) | Lightning, 1 노드 × 4 GPU | 2~4분 | ≈ 0.45 ~ 0.55 | 50~80% |
| 05 (M×N) | Lightning, 2 노드 × 4 GPU | 2~3분 | ≈ 0.45 ~ 0.55 | 40~70% |

체크리스트:
- `val/loss` 가 첫 epoch 직후 0.6+ → 학습 자체는 진행 중. 5~7 epoch 안에 0.5 아래로 떨어져야 정상.
- early stop이 epoch 5 이내에 발동 → patience(3) 대비 너무 빠르면 `min_delta` 가 큼 (현재 1e-4).
- multi-node throughput 이 single-node와 비슷 → inter-node bandwidth 병목 ([`common-pitfalls.md §10`](../00-foundations/common-pitfalls.md)).
- GPU util 이 10% 미만 → DataLoader 병목. `num_workers` / `pin_memory` 점검.
- MLflow UI System Metrics 탭에서 GPU 차트가 비어 있음 → `nvidia-ml-py` 미설치 또는 multi-node에서 driver-side만 켠 경우 ([`common-pitfalls.md §9`](../00-foundations/common-pitfalls.md)).

## ⚠️ 제약

- HuggingFace Accelerate는 CLI 기반 launcher라 노트북 only 방식에 적합하지 않습니다. 단일 노트북에서 가시 GPU 수를 자동 감지해 1×1/1×N/M×N을 모두 다루는 [`02-script-based/08-launch_accelerator_MxN.ipynb`](../02-script-based/08-launch_accelerator_MxN.ipynb) 참고.
- 멀티 노드 (03, 05) 노트북에서는 driver→worker로 `db_host`/`db_token`을 명시 전달해야 합니다. [`00-foundations/common-pitfalls.md#2-1`](../00-foundations/common-pitfalls.md)
- Serverless GPU는 multi-node를 지원하지 않습니다. 03, 05는 반드시 Classic GPU 클러스터에서 실행 ([`databricks-environments.md`](../00-foundations/databricks-environments.md)).

## ➡️ 다음

스크립트 분리 패턴: [`02-script-based/`](../02-script-based/README.md)
