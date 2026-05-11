# 01-3 · 노트북 only · M×N GPU

> 한 노트북 안에서 `TorchDistributor.run(fn, ...)`로 다중 노드 × 다중 GPU 분산학습을 시작한다. 병렬화는 multi-node DDP.

## 🎯 시나리오

- Two-Tower MLP 대형 변형 (~수 GB params, [recommender-baseline](../../00-foundations/recommender-baseline.md))
- 데이터 처리량을 노드 수로 확장하고 싶을 때
- 코드 분리 없이 노트북에서 multi-node를 시도하는 단계

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (user 10M × item 5M, emb dim 256) |
| 라이브러리 | `torch`, `mlflow` |
| 병렬화 | **DDP** (multi-node) |
| 데이터 | 합성 interaction → parquet shard (UC Volume) |
| 실행 | `TorchDistributor(num_processes=M*N, local_mode=False).run(fn, ...)` |
| 추적 | MLflow autolog (rank 0만) |
| 목표 학습 시간 | 15분 |

## 🖥️ 클러스터 권장 사양

| 클라우드 | 인스턴스 | 노드 수 | GPU/노드 |
|---------|---------|--------|---------|
| AWS | `g5.12xlarge` | 2 | 4× A10G 24GB |
| AWS (대형) | `p4d.24xlarge` | 2~4 | 8× A100 40GB |

- DBR: 15.x ML GPU
- Cluster type: **Classic GPU 클러스터** (Serverless/AI Runtime은 multi-node 미지원)
- Autoscaling: **끈다**

## 📂 파일

```
03-multi-node-multi-gpu/
├── README.md
├── 00_setup.py
├── 01_data_prep.py         # 합성 interaction → parquet shard
├── 02_train_ddp.py
└── 03_eval_and_register.py
```

## 🚀 실행 순서

1. `00_setup.py` 실행.
2. `01_data_prep.py` 실행 → 합성 interaction을 parquet shard로 UC Volume에 저장.
3. `02_train_ddp.py` 실행 → driver에서 `TorchDistributor.run(train_fn, ...)` 호출.
4. `03_eval_and_register.py` 실행.

## 🧬 핵심 패턴

```python
from pyspark.ml.torch.distributor import TorchDistributor

NUM_NODES = 2
GPUS_PER_NODE = 4
NUM_PROCESSES = NUM_NODES * GPUS_PER_NODE

def train_fn(run_id, data_path, ckpt_dir, n_users, n_items, emb_dim):
    # 모든 import는 함수 내부에서
    import os, mlflow, torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    # ... Two-Tower MLP + DistributedSampler 학습 루프 ...

with mlflow.start_run(run_name="recommender-mxn-ddp"):
    TorchDistributor(
        num_processes=NUM_PROCESSES,
        local_mode=False,
        use_gpu=True,
    ).run(train_fn, run_id=..., data_path=..., ckpt_dir=...,
          n_users=N_USERS, n_items=N_ITEMS, emb_dim=EMB_DIM)
```

## ⚠️ 함정

- `local_mode=False` 필수. True로 두면 driver 한 노드에서만 돈다.
- `train_fn`은 모듈 최상위 또는 다른 파일에서 정의해야 pickle이 안전 ([common-pitfalls #2](../../00-foundations/common-pitfalls.md#2-pickle-에러-torchdistributorrunfn-)).
- 체크포인트는 `/local_disk0/`에 저장 후 종료 시 UC Volume copy ([uc-volumes](../../00-foundations/uc-volumes-checkpoints.md)).
- 작은 모델에서 multi-node throughput 한계 ([common-pitfalls #9](../../00-foundations/common-pitfalls.md#9-multi-node-ddp인데-throughput이-안-올라간다)).

## ➡️ 다음 셀

- 옆: 매트릭스의 마지막 열 → 끝.
- 아래 (코드를 파일로 분리해 운영화): [02-3 · 스크립트 · M×N GPU](../../02-script-based/03-multi-node-multi-gpu/)

## 📚 출처/참고

- TorchDistributor: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor
- 기준 모델 정의: [`00-foundations/recommender-baseline.md`](../../00-foundations/recommender-baseline.md)
