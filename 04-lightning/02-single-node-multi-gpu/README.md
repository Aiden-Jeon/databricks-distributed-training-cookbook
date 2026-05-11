# 04-2 · Lightning · 1×N GPU

> `Trainer(devices=N, strategy="ddp")`로 단일 노드 다중 GPU 분산학습. Lightning이 launcher를 직접 처리한다.

## 🎯 시나리오

- 04-1에서 LightningModule을 작성한 뒤 같은 노드의 GPU를 다 활용
- Lightning 인터페이스를 유지하면서 DDP로 확장

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (user 1M × item 500K, emb dim 128) |
| 라이브러리 | `pytorch-lightning`, `torch` |
| 병렬화 | **DDP** |
| 데이터 | 합성 interaction → `LightningDataModule` |
| 실행 | `Trainer(devices=N, strategy="ddp")` 호출, driver는 노트북 |
| 추적 | `MLFlowLogger` |

## 🖥️ 클러스터 권장 사양

[01-2와 동일](../../01-notebook-only/02-single-node-multi-gpu/README.md#️-클러스터-권장-사양).

## 📂 파일

```
02-single-node-multi-gpu/
├── README.md
├── driver_notebook.py
└── train.py
```

## 🚀 실행 순서

1. driver에서 `devices` 결정.
2. `train.py` 실행 → Lightning이 자체적으로 N개의 worker 프로세스를 띄움.
3. driver에서 평가·등록.

## 🧬 핵심 패턴

```python
trainer = pl.Trainer(
    devices=4,
    accelerator="gpu",
    strategy="ddp",
    precision="bf16-mixed",
    logger=MLFlowLogger(...),
)
trainer.fit(module, datamodule=dm)
```

## ⚠️ 함정

- Databricks 노트북에서 Lightning의 `ddp_spawn`은 잘 안 됨 → `strategy="ddp"` (fork 기반) 사용.
- MLflow 로깅은 Lightning이 rank 0에서만 처리 → 별도 가드 불필요.
- DataModule의 `train_dataloader`는 일반 DataLoader로 두면 Lightning이 `DistributedSampler`를 자동으로 끼워준다.

## ➡️ 다음 셀

- 옆: [04-3 · Lightning · M×N GPU](../03-multi-node-multi-gpu/)
- 위: [03-2](../../03-cli-accelerate/02-single-node-multi-gpu/), [02-2](../../02-script-based/02-single-node-multi-gpu/)
- 아래: 행의 끝.

## 📚 출처/참고

- Lightning DDP: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
