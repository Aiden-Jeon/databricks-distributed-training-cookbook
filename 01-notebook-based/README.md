# 01 · Notebook-based

> 모든 학습 코드를 노트북 셀 안에 두는, 가장 빠른 PoC 경로입니다.

## 🧭 노트북 흐름

번호 순서대로 실행하는 흐름입니다.

| 순서 | 파일 | 역할 | 클러스터 |
|------|------|------|----------|
| 00 | [`00-setup.ipynb`](00-setup.ipynb) | 패키지 설치, UC 경로, 단일 `CONFIG`, MLflow experiment, `DB_HOST/DB_TOKEN` | 아무 클러스터(코드 실행만) |
| 01 | [`01-data_prep.ipynb`](01-data_prep.ipynb) | MovieLens 25M 다운로드 → implicit feedback 변환 → UC Volume parquet shard | 단일 노드 CPU도 가능 |
| 02 | [`02-train_torch_distributor_1x1_1xN.ipynb`](02-train_torch_distributor_1x1_1xN.ipynb) | TorchDistributor 1×1 / 1×N 섹션 | Single node GPU |
| 03 | [`03-train_torch_distributor_MxN.ipynb`](03-train_torch_distributor_MxN.ipynb) | TorchDistributor M×N | Multi node GPU |
| 04 | [`04-train_pytorch_lightning_1x1_1xN.ipynb`](04-train_pytorch_lightning_1x1_1xN.ipynb) | Lightning 1×1 / 1×N 섹션 | Single node GPU |
| 05 | [`05-train_pytorch_lightning_MxN.ipynb`](05-train_pytorch_lightning_MxN.ipynb) | Lightning M×N | Multi node GPU |

Single-node 노트북(02, 04)은 같은 single-node 클러스터에서 둘 다 실행할 수 있습니다. Multi-node 노트북(03, 05)은 multi-node 클러스터에서 실행합니다. autoscaling·single-node 토글 설정이 서로 달라, 두 클러스터를 분리해서 띄우는 것을 권장합니다.

## 🔀 매트릭스

launcher와 토폴로지의 조합이 어느 노트북·호출 패턴에 매핑되는지 보여 줍니다.

| | TorchDistributor | Lightning |
|----|---|---|
| **1×1 GPU** | 02 (`TorchDistributor(num_processes=1, local_mode=True).run(fn)` + DDP) | 04 (`Trainer(devices=1)`) |
| **1×N GPU** | 02 (`TorchDistributor(num_processes=N, local_mode=True).run(fn)` + DDP) | 04 (`TorchDistributor(local_mode=True).run(fn)` + `Trainer(strategy='ddp')`) |
| **M×N GPU** | 03 (`TorchDistributor(num_processes=M*N, local_mode=False).run(fn)` + DDP) | 05 (TorchDistributor + `Trainer(devices=N, num_nodes=M, strategy='ddp')`) |

## 🖥️ 클러스터 세팅

같은 학습 launcher라도 single-node와 multi-node 노트북은 **별도 클러스터**에서 띄우는 것을 권장합니다. autoscaling·single-node 토글 설정이 다르기 때문입니다. 상세 인스턴스 비교는 [`00-foundations/env-databricks-environments.md`](../00-foundations/env-databricks-environments.md)를 참고하세요.

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

클러스터를 만들 때 자주 놓치는 지점을 정리하면 다음과 같습니다.

- **Single node 토글**: 02·04는 ON, 03·05는 OFF로 둡니다. 같은 클러스터를 토글만 바꿔 재사용할 수는 없으며, 재생성이 필요합니다.
- **Autoscaling**: 03·05는 반드시 OFF로 둡니다. DDP는 학습 시작 시 노드 수가 고정되어야 합니다.
- **Driver도 GPU**: M×N에서 TorchDistributor 코디네이션은 driver에서 일어납니다. CPU driver로 두면 일부 PyTorch import가 실패합니다.
- **DBR 17.3 LTS ML**: 일반 DBR이 아닌 "Machine Learning" 런타임을 선택해야 `torch`와 CUDA가 사전 설치되어 있습니다. GPU 노드 타입(`g5.*` 등)을 선택하면 GPU 이미지가 자동 적용됩니다.
- 02의 1×1 섹션은 GPU 1개만 사용해도 충분하므로 `g5.2xlarge`(1× A10G) 클러스터에서 따로 돌려도 됩니다. 1×N과 클러스터를 공유하려면 `g5.12xlarge`를 그대로 쓰면 됩니다.

H100/A100을 쓰려면 `p4d.24xlarge` 또는 H100 인스턴스로 교체합니다 ([`env-databricks-environments.md`](../00-foundations/env-databricks-environments.md)).

## 📊 MLflow 로깅

토폴로지별 로깅 동작이 미묘하게 다르므로 짚어 둡니다.

- 모든 토폴로지(1×1 / 1×N / M×N)의 run은 `00-setup`에서 만든 단일 experiment(`/Users/<email>/recommender-notebook-based`)에 누적됩니다. 같은 experiment 안에서 run을 비교하세요.
- **System metrics**(GPU 활용도, 메모리, I/O)는 `mlflow.start_run(..., log_system_metrics=True)`로 수집됩니다.
  - Single-node(02, 04): driver와 worker가 같은 머신이므로 driver-side에서만 켜도 충분합니다.
  - **Multi-node(03, 05)**: driver는 학습에 참여하지 않으므로 driver-side만 켜면 idle 메트릭만 잡힙니다. 03·05의 학습 함수는 **rank-0 worker에서 `mlflow.start_run(run_id=..., log_system_metrics=True)`로 attach**해 worker 노드의 GPU 메트릭까지 함께 기록합니다 ([`00-foundations/ops-mlflow-tracking.md §1`](../00-foundations/ops-mlflow-tracking.md)).

## 📈 기대 결과

ML-25M 데이터(positive+negative ≈ 25M 행)와 `max_steps_per_epoch=200` 기준입니다. 환경(NCCL 버전, EFA 유무, 디스크 캐시 상태)에 따라 ±30% 정도 편차는 흔합니다.

| 노트북 | 토폴로지 | 학습 시간 | val/loss (10 epoch 또는 early stop) | GPU util (rank-0 기준) |
|--------|---------|----------|-----------------------------------|----------------------|
| 02 (1×1) | 1 노드 × 1 GPU (A10G) | 3~6분 | ≈ 0.45 ~ 0.55 | 60~85% |
| 02 (1×N) | 1 노드 × 4 GPU (g5.12xlarge) | 2~4분 | ≈ 0.45 ~ 0.55 | 50~80% (DataLoader 병목 시 낮음) |
| 03 (M×N) | 2 노드 × 4 GPU | 2~3분 | ≈ 0.45 ~ 0.55 | 40~70% (EFA 없으면 낮음) |
| 04 (1×1) | Lightning, 1 노드 × 1 GPU | 3~6분 | ≈ 0.45 ~ 0.55 | 60~85% |
| 04 (1×N) | Lightning, 1 노드 × 4 GPU | 2~4분 | ≈ 0.45 ~ 0.55 | 50~80% |
| 05 (M×N) | Lightning, 2 노드 × 4 GPU | 2~3분 | ≈ 0.45 ~ 0.55 | 40~70% |

실행이 끝나면 다음 신호를 차례로 점검하세요.

- `val/loss`가 첫 epoch 직후 0.6 이상이라도 학습 자체는 진행 중입니다. 5~7 epoch 안에 0.5 아래로 떨어져야 정상입니다.
- early stop이 epoch 5 이내에 발동하면 patience(3) 대비 너무 빠른 것이며, `min_delta`(현재 1e-4)가 크게 잡혀 있을 가능성이 높습니다.
- multi-node throughput이 single-node와 비슷하다면 inter-node bandwidth 병목입니다 ([`debug-common-pitfalls.md §10`](../docs/debug-common-pitfalls.md)).
- GPU util이 10% 미만이면 DataLoader 병목이므로 `num_workers`와 `pin_memory`를 점검합니다.
- MLflow UI의 System Metrics 탭에서 GPU 차트가 비어 있다면 `nvidia-ml-py`가 미설치이거나 multi-node에서 driver-side만 켠 경우입니다 ([`debug-common-pitfalls.md §9`](../docs/debug-common-pitfalls.md)).

## ⚠️ 제약

본 행에서 의도적으로 비워 둔 부분이 있어 정리해 둡니다.

- HuggingFace Accelerate는 CLI 기반 launcher라 노트북 only 방식에 적합하지 않습니다. 단일 노트북에서 가시 GPU 수를 자동 감지해 1×1/1×N/M×N을 모두 다루는 [`02-script-based/08-launch_accelerator_MxN.ipynb`](../02-script-based/08-launch_accelerator_MxN.ipynb)을 참고하세요.
- 멀티 노드(03, 05) 노트북에서는 driver에서 worker로 `db_host`와 `db_token`을 명시 전달해야 합니다 ([`docs/debug-common-pitfalls.md#2-1`](../docs/debug-common-pitfalls.md)).
- Serverless GPU는 multi-node를 지원하지 않습니다. 03, 05는 반드시 Classic GPU 클러스터에서 실행하세요 ([`env-databricks-environments.md`](../00-foundations/env-databricks-environments.md)).

## ➡️ 다음

스크립트 분리 패턴은 [`02-script-based/`](../02-script-based/README.md)에서 이어집니다.
