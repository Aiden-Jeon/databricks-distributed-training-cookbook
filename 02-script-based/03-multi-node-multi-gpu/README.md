# 02-3 · 스크립트 · M×N GPU

> 같은 `train.py`를 `TorchDistributor.run("train.py", *args)`로 multi-node에서 DDP 실행한다. 본 쿡북 스크립트 트랙의 최종 형태.

## 🎯 시나리오

- 02-2의 학습 스크립트를 검증한 뒤 노드 수만 늘리는 단계
- 추천 도메인의 거대 interaction 데이터를 노드 수에 비례하게 처리하고 싶을 때
- 같은 코드로 1×N → M×N 전환을 검증하고 싶을 때

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (user 10M × item 5M, emb dim 256) |
| 라이브러리 | `torch`, `mlflow` |
| 병렬화 | **DDP** (multi-node) |
| 데이터 | parquet shard (UC Volume) |
| 실행 | `TorchDistributor(num_processes=M*N, local_mode=False).run("train.py", *args)` |
| 추적 | MLflow autolog (rank 0) |

## 🖥️ 클러스터 권장 사양

[01-3과 동일](../../01-notebook-only/03-multi-node-multi-gpu/README.md#️-클러스터-권장-사양). Classic GPU 클러스터 필수.

## 📂 파일

```
03-multi-node-multi-gpu/
├── README.md
├── driver_notebook.py
├── train.py                 # entrypoint, argparse
└── configs/
    └── training_args.yaml
```

## 🚀 실행 순서

1. `driver_notebook.py`에서 NUM_NODES·GPUS_PER_NODE 변수 설정.
2. `TorchDistributor.run("train.py", "--config", cfg_path, "--run-id", run_id)` 호출.
3. driver에서 평가·등록.

## 🧬 핵심 패턴

```python
from pyspark.ml.torch.distributor import TorchDistributor

NUM_PROCESSES = NUM_NODES * GPUS_PER_NODE
with mlflow.start_run(run_name="recommender-mxn-ddp") as run:
    TorchDistributor(
        num_processes=NUM_PROCESSES,
        local_mode=False,
        use_gpu=True,
    ).run("train.py", "--config", "configs/training_args.yaml",
                     "--run-id", run.info.run_id)
```

## ⚠️ 함정

- `train.py`가 인자만으로 자기 자신을 구동할 수 있어야 한다 (notebook 의존 X).
- 학습 끝난 후 체크포인트는 `/local_disk0/` → UC Volume copy ([uc-volumes](../../00-foundations/uc-volumes-checkpoints.md)).
- parquet shard 경로는 driver에서 미리 생성·검증 후 인자로 넘긴다.
- 작은 모델 + 다수 노드 조합에서 throughput 한계 ([common-pitfalls #9](../../00-foundations/common-pitfalls.md#9-multi-node-ddp인데-throughput이-안-올라간다)).

## ➡️ 다음 셀

- 옆: 마지막 열.
- 위: [02-2](../02-single-node-multi-gpu/), [01-3](../../01-notebook-only/03-multi-node-multi-gpu/)
- 아래: [03-3 · Accelerate · M×N GPU](../../03-cli-accelerate/03-multi-node-multi-gpu/)

## 📚 출처/참고

- TorchDistributor: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor
- 기준 모델 정의: [`00-foundations/recommender-baseline.md`](../../00-foundations/recommender-baseline.md)
