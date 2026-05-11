# 03-1 · Accelerate · 1×1 GPU

> HuggingFace `accelerate launch` CLI로 학습을 시작한다. 1×1 GPU에서는 가장 단순한 single-process 구성.

## 🎯 시나리오

- HF 에코시스템 표준 launcher(`accelerate launch`)를 그대로 따르고 싶은 경우
- DDP는 다음 단계, 우선 single-process로 검증

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (01-1과 동일, [recommender-baseline](../../00-foundations/recommender-baseline.md)) |
| 라이브러리 | `torch`, `accelerate`, `mlflow` |
| 병렬화 | 단일 GPU |
| 데이터 | 합성 interaction → Delta → DataLoader |
| 실행 | `%sh accelerate launch --num_processes 1 --config_file configs/accelerate_single_gpu.yaml train.py` |
| 추적 | MLflow autolog |

## 🖥️ 클러스터 권장 사양

[01-1과 동일](../../01-notebook-only/01-single-node-single-gpu/README.md#️-클러스터-권장-사양).

## 📂 파일

```
01-single-node-single-gpu/
├── README.md
├── driver_notebook.py
├── train.py                 # accelerate launch entrypoint
└── configs/
    └── accelerate_single_gpu.yaml
```

## 🚀 실행 순서

1. `driver_notebook.py`에서 cfg 경로 설정.
2. `%sh accelerate launch --config_file configs/accelerate_single_gpu.yaml train.py --config configs/training_args.yaml`
3. driver에서 등록.

## 🧬 핵심 패턴

```yaml
# configs/accelerate_single_gpu.yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
num_processes: 1
mixed_precision: bf16
```

```python
# train.py
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
for batch in loader:
    ...
```

## ⚠️ 함정

- `accelerate config`로 만든 default config는 `~/.cache/huggingface/accelerate/...`에 저장되며 클러스터 재시작 시 사라진다 → 본 쿡북은 **config 파일을 repo 안에 둔다**.
- `%sh`에서 실행한 프로세스는 driver의 Python 환경을 본다. `%pip install`이 먼저.

## ➡️ 다음 셀

- 옆: [03-2 · Accelerate · 1×N GPU](../02-single-node-multi-gpu/)
- 위: [02-1](../../02-script-based/01-single-node-single-gpu/)
- 아래: [04-1 · Lightning · 1×1 GPU](../../04-lightning/01-single-node-single-gpu/)

## 📚 출처/참고

- HF Accelerate 문서: https://huggingface.co/docs/accelerate/index
