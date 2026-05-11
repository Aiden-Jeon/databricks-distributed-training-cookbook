# 01 · Notebook-based

> 모든 학습 코드가 노트북 셀 안에 있다. 가장 빠른 PoC 경로.

## 🧭 노트북 흐름

| 순서 | 파일 | 역할 |
|------|------|------|
| 00 | [`00-setup.ipynb`](00-setup.ipynb) | 패키지 설치, 환경 변수, UC 경로, GPU 개수 정의 |
| 01 | [`01-data_prep.ipynb`](01-data_prep.ipynb) | 합성 user-item interaction 데이터 생성 → Delta/Volume |
| 02 | [`02-train_torch_distributor.ipynb`](02-train_torch_distributor.ipynb) | TorchDistributor로 학습. 토폴로지별 섹션 (1×1 / 1×N / M×N) |
| 03 | [`03-train_pytorch_lightning.ipynb`](03-train_pytorch_lightning.ipynb) | Lightning `Trainer`로 학습. 토폴로지별 섹션 (1×1 / 1×N / M×N) |

## 🔀 매트릭스

학습 노트북 안에서 토폴로지를 섹션으로 비교한다.

| | TorchDistributor | Lightning |
|----|---|---|
| **1×1 GPU** | standard PyTorch loop | `Trainer(devices=1, num_nodes=1)` |
| **1×N GPU** | `TorchDistributor(num_processes=N, local_mode=True).run(fn, ...)` + DDP | `Trainer(devices=N, num_nodes=1, strategy="ddp")` |
| **M×N GPU** | `TorchDistributor(num_processes=M*N, local_mode=False).run(fn, ...)` + DDP | `Trainer(devices=N, num_nodes=M, strategy="ddp")` |

## ⚠️ 제약

- HuggingFace Accelerate는 CLI 기반 launcher라 노트북 only 방식에 적합하지 않다. [`02-script-based/04-launch_accelerator.ipynb`](../02-script-based/04-launch_accelerator.ipynb) 참고.
- 멀티 노드 (M×N) 셀에서는 driver→worker로 `db_host`/`db_token`을 명시 전달해야 한다. [`00-foundations/common-pitfalls.md#2-1`](../00-foundations/common-pitfalls.md)

## ➡️ 다음

스크립트 분리 패턴: [`02-script-based/`](../02-script-based/)
