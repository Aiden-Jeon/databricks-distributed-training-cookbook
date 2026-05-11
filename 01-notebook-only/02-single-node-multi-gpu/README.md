# 01-2 · 노트북 only · 1×N GPU

> 한 노트북에서 `@distributed` 데코레이터로 학습 함수를 감싸 단일 노드의 N개 GPU를 동시에 사용한다. 병렬화는 DDP만.

## 🎯 시나리오

- Two-Tower MLP 중형 변형 (~200M params, [recommender-baseline](../../00-foundations/recommender-baseline.md))
- 단일 인스턴스의 4~8 GPU를 풀로 활용하여 데이터 처리량 확장
- 노트북 환경을 유지하면서 multi-GPU 분산학습이 필요한 경우

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (user 1M × item 500K, emb dim 128) |
| 라이브러리 | `torch`, `mlflow` |
| 병렬화 | **DDP** (DistributedDataParallel) |
| 데이터 | 합성 interaction → Delta → `DistributedSampler` |
| 실행 | `@distributed(gpus=N)`로 함수 래핑 후 호출 |
| 추적 | MLflow autolog (rank 0만) |
| 목표 학습 시간 | 10분 |

## 🖥️ 클러스터 권장 사양

| 클라우드 | 인스턴스 | GPU |
|---------|---------|-----|
| AWS | `g5.12xlarge` | 4× A10G 24GB |
| AWS (큰 변형) | `p4d.24xlarge` | 8× A100 40GB |

- DBR: 15.x ML GPU **또는 AI Runtime** (Serverless GPU Compute)
- `@distributed` 데코레이터는 AI Runtime에서 가장 매끄럽게 동작

## 📂 파일

```
02-single-node-multi-gpu/
├── README.md
├── 00_setup.py
├── 01_data_prep.py
├── 02_train_ddp.py          # @distributed + DDP
└── 03_eval_and_register.py
```

## 🚀 실행 순서

1. `00_setup.py` 실행 → 패키지·경로 설정, `GPUS_PER_NODE` 결정.
2. `01_data_prep.py` 실행 → 합성 interaction Delta 테이블 생성.
3. `02_train_ddp.py` 실행 → `@distributed(gpus=N)` 학습 함수 호출.
4. `03_eval_and_register.py` 실행 → AUC 측정·등록.

## 🧬 핵심 패턴

```python
from databricks.ml.distributed import distributed

@distributed(gpus=4, gpu_resource="a10g")  # AI Runtime 시그니처
def train_fn(run_id: str, data_path: str, ckpt_dir: str,
             n_users: int, n_items: int, emb_dim: int):
    import os, mlflow, torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    rank = int(os.environ["RANK"])
    if rank == 0:
        mlflow.start_run(run_id=run_id)

    model = TwoTowerMLP(n_users, n_items, emb_dim).cuda()
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    # ... DistributedSampler + 학습 루프 ...

with mlflow.start_run(run_name="recommender-1xn-ddp") as run:
    train_fn(run.info.run_id, DATA_PATH, CKPT_DIR, N_USERS, N_ITEMS, EMB_DIM)
```

## ⚠️ 함정

- `@distributed` 함수 안에서 driver 변수를 캡처하면 pickle 에러가 난다 → 인자로 전달 ([common-pitfalls #2](../../00-foundations/common-pitfalls.md#2-pickle-에러-torchdistributorrunfn-)).
- MLflow는 rank 0에서만 시작 ([mlflow-tracking](../../00-foundations/mlflow-tracking.md)).
- `DistributedSampler(..., drop_last=True)`로 rank 간 step 수를 맞춘다.

## ➡️ 다음 셀

- 옆: [01-3 · 노트북 · M×N GPU](../03-multi-node-multi-gpu/)
- 아래: [02-2 · 스크립트 · 1×N GPU](../../02-script-based/02-single-node-multi-gpu/)

## 📚 출처/참고

- Multi-GPU workload (`@distributed`): https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training
- 기준 모델 정의: [`00-foundations/recommender-baseline.md`](../../00-foundations/recommender-baseline.md)
