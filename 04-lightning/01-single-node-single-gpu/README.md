# 04-1 · Lightning · 1×1 GPU

> PyTorch Lightning `Trainer`로 학습을 구성한다. 1×1 GPU에서는 `Trainer(devices=1)` 기본 사용.

## 🎯 시나리오

- 학습 루프·콜백을 Lightning이 관리하길 원할 때
- 동일한 LightningModule을 04-2(DDP) · 04-3(multi-node DDP)로 확장하기 위한 시작점

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (01-1과 동일, [recommender-baseline](../../00-foundations/recommender-baseline.md)) |
| 라이브러리 | `pytorch-lightning`, `torch` |
| 병렬화 | 단일 GPU |
| 데이터 | 합성 interaction → `LightningDataModule` |
| 실행 | 노트북에서 `trainer.fit(module, datamodule=...)` |
| 추적 | `MLFlowLogger` |

## 🖥️ 클러스터 권장 사양

[01-1과 동일](../../01-notebook-only/01-single-node-single-gpu/README.md#️-클러스터-권장-사양).

## 📂 파일

```
01-single-node-single-gpu/
├── README.md
├── driver_notebook.py
└── train.py                 # LightningModule + DataModule + Trainer
```

## 🚀 실행 순서

1. driver에서 cfg 변수 설정.
2. `%run ./train.py` 또는 `!python train.py`.
3. driver에서 등록.

## 🧬 핵심 패턴

```python
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger


class RecommenderModule(pl.LightningModule):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.model = TwoTowerMLP(n_users, n_items, emb_dim)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        users, items, labels = batch
        logits = self.model(users, items)
        loss = self.loss_fn(logits, labels)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    logger=MLFlowLogger(experiment_name="...", tracking_uri="databricks"),
    max_epochs=1,
)
trainer.fit(RecommenderModule(...), datamodule=RecsysDataModule(...))
```

## ⚠️ 함정

- `precision="bf16-mixed"`는 Ampere+에서만. T4는 `"16-mixed"`.
- Lightning은 자체적으로 체크포인트를 저장 → `ModelCheckpoint(dirpath="/local_disk0/...")` 권장.

## ➡️ 다음 셀

- 옆: [04-2 · Lightning · 1×N GPU](../02-single-node-multi-gpu/)
- 위: [03-1](../../03-cli-accelerate/01-single-node-single-gpu/), [02-1](../../02-script-based/01-single-node-single-gpu/)
- 아래: 행의 끝.

## 📚 출처/참고

- Lightning docs: https://lightning.ai/docs/pytorch/stable/
