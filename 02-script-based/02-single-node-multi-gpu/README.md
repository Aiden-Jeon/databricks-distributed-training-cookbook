# 02-2 · 스크립트 · 1×N GPU

> `train.py`로 학습 코드를 분리한 뒤, 노트북에서 `@distributed` 내부 entrypoint로 호출해 단일 노드의 N GPU에서 DDP로 돌린다.

## 🎯 시나리오

- 01-2와 동일한 토폴로지, 단 코드가 노트북이 아닌 파일에 있는 형태
- 같은 `train.py`를 1×N(여기) → M×N(02-3)로 재사용

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (user 1M × item 500K, emb dim 128) |
| 라이브러리 | `torch`, `mlflow` |
| 병렬화 | **DDP** |
| 데이터 | 합성 interaction → Delta / parquet |
| 실행 | `@distributed` 내부에서 `train.main(cfg, run_id)` 호출 |
| 추적 | MLflow autolog (rank 0) |

## 🖥️ 클러스터 권장 사양

[01-2와 동일](../../01-notebook-only/02-single-node-multi-gpu/README.md#️-클러스터-권장-사양).

## 📂 파일

```
02-single-node-multi-gpu/
├── README.md
├── driver_notebook.py
├── train.py
└── configs/
    └── training_args.yaml
```

## 🚀 실행 순서

1. `driver_notebook.py`에서 cfg 경로·run name 변수 설정.
2. `@distributed(gpus=N)` 함수 내부에서 `train.main(cfg, run_id)` 호출.
3. driver에서 평가·등록.

## 🧬 핵심 패턴

```python
# driver_notebook.py
from databricks.ml.distributed import distributed

@distributed(gpus=4)
def launch(cfg_path: str, run_id: str):
    import train
    train.main(cfg_path, run_id=run_id)

with mlflow.start_run(run_name="recommender-2xn-ddp") as run:
    launch("configs/training_args.yaml", run_id=run.info.run_id)
```

## ⚠️ 함정

- `train.py`가 `dbutils`/`spark`에 의존하지 않도록 한다 → 인자로 받은 경로/하이퍼파라미터만 본다.
- DDP는 `DistributedSampler(..., drop_last=True)`로 rank 간 step 수를 맞춘다.

## ➡️ 다음 셀

- 옆: [02-3 · 스크립트 · M×N GPU](../03-multi-node-multi-gpu/)
- 위: [01-2](../../01-notebook-only/02-single-node-multi-gpu/)
- 아래: [03-2 · Accelerate · 1×N GPU](../../03-cli-accelerate/02-single-node-multi-gpu/)

## 📚 출처/참고

- `@distributed` 문서: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training
