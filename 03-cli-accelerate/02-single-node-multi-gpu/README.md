# 03-2 · Accelerate · 1×N GPU

> `accelerate launch --num_processes N`으로 단일 노드 다중 GPU를 DDP로 분산학습한다.

## 🎯 시나리오

- 03-1을 검증한 뒤 같은 노드의 GPU를 다 쓰고 싶을 때
- HF Accelerate launcher를 유지하면서 multi-GPU로 확장

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (user 1M × item 500K, emb dim 128) |
| 라이브러리 | `torch`, `accelerate`, `mlflow` |
| 병렬화 | **DDP** (`accelerate_ddp.yaml`) |
| 데이터 | 합성 interaction → Delta / parquet |
| 실행 | `%sh accelerate launch --config_file configs/accelerate_ddp.yaml train.py` |
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
    └── accelerate_ddp.yaml
```

## 🚀 실행 순서

1. driver에서 cfg·N 결정.
2. `%sh accelerate launch --num_processes 4 --config_file configs/accelerate_ddp.yaml train.py`
3. driver에서 등록.

## 🧬 핵심 패턴

```yaml
# configs/accelerate_ddp.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
mixed_precision: bf16
```

```python
# train.py
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
# Accelerate가 DDP wrap과 sampler 분할을 자동 처리
for batch in loader:
    ...
    accelerator.backward(loss)
    optimizer.step()
```

## ⚠️ 함정

- `accelerate launch`로 띄운 프로세스는 driver 노트북의 `os.environ`을 상속한다 → `MLFLOW_EXPERIMENT_NAME`을 driver에서 미리 export.
- DataLoader를 `accelerator.prepare()`에 통과시키면 `DistributedSampler`가 자동 적용된다 → 직접 sampler를 만들지 않아도 된다.

## ➡️ 다음 셀

- 옆: [03-3 · Accelerate · M×N GPU](../03-multi-node-multi-gpu/)
- 위: [02-2](../../02-script-based/02-single-node-multi-gpu/), [01-2](../../01-notebook-only/02-single-node-multi-gpu/)
- 아래: [04-2 · Lightning · 1×N GPU](../../04-lightning/02-single-node-multi-gpu/)

## 📚 출처/참고

- HF Accelerate 분산 학습: https://huggingface.co/docs/accelerate/basic_tutorials/launch
