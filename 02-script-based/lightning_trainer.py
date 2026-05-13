"""PyTorch Lightning 학습 모듈과 fit() 진입점.

- 04-launch_lightning_trainer_single_node.ipynb의 1x1 섹션은 driver에서 직접 fit() 호출.
- 1xN / MxN 섹션은 TorchDistributor.run(fit, **kwargs) 형태로 호출. multi-node 워커 프로세스는
  fresh Python으로 시작하므로 script_dir을 sys.path에 추가해 sibling 모듈(model.py)을 import.

토폴로지(1x1 / 1xN / MxN)는 호출 측에서 devices, num_nodes만 바꿔 전환된다.
"""

import os
import sys


def fit(
    experiment_path,
    run_id,
    db_host,
    db_token,
    data_dir,
    ckpt_dir,
    n_users,
    n_items,
    emb_dim,
    tower_hidden,
    batch_size,
    num_epochs,
    max_steps_per_epoch,
    patience,
    min_delta,
    devices,
    num_nodes,
    topology,
    script_dir,
):
    """Lightning Trainer.fit() 호출. TorchDistributor child 또는 driver에서 직접 호출.

    1x1에서는 strategy를 'auto'로 두고, devices>1 또는 num_nodes>1일 때만 'ddp'를 강제한다.
    """
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import lightning as L
    import mlflow
    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import MLFlowLogger
    from torch.utils.data import DataLoader, TensorDataset

    from model import TwoTowerMLP

    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token

    class TwoTowerLitModule(L.LightningModule):
        def __init__(
            self, n_users, n_items, emb_dim, tower_hidden, lr=1e-3, weight_decay=1e-5
        ):
            super().__init__()
            self.save_hyperparameters()
            self.model = TwoTowerMLP(n_users, n_items, emb_dim, tower_hidden)
            self.loss_fn = nn.BCEWithLogitsLoss()

        def forward(self, user_ids, item_ids):
            return self.model(user_ids, item_ids)

        def training_step(self, batch, batch_idx):
            u, i, y = batch
            logits = self(u, i)
            loss = self.loss_fn(logits, y)
            self.log(
                "train/loss",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )
            return loss

        def validation_step(self, batch, batch_idx):
            u, i, y = batch
            logits = self(u, i)
            loss = self.loss_fn(logits, y)
            # EarlyStopping이 monitor하는 키. on_epoch=True + sync_dist=True로 DDP rank 합산.
            self.log(
                "val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

    class InteractionsDataModule(L.LightningDataModule):
        def __init__(self, data_dir, batch_size, num_workers=2):
            super().__init__()
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.num_workers = num_workers

        def _load_split(self, split):
            split_dir = os.path.join(self.data_dir, split)
            files = sorted(
                os.path.join(split_dir, f)
                for f in os.listdir(split_dir)
                if f.endswith(".parquet")
            )
            table = pq.read_table(files, columns=["user_id", "item_id", "label"])
            return TensorDataset(
                torch.from_numpy(table.column("user_id").to_numpy()),
                torch.from_numpy(table.column("item_id").to_numpy()),
                torch.from_numpy(table.column("label").to_numpy()),
            )

        def setup(self, stage=None):
            self.train_dataset = self._load_split("train")
            self.val_dataset = self._load_split("val")

        def train_dataloader(self):
            # Trainer가 strategy='ddp'일 때 DistributedSampler를 자동 주입.
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        def val_dataloader(self):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

    model = TwoTowerLitModule(
        n_users=n_users, n_items=n_items, emb_dim=emb_dim, tower_hidden=tower_hidden
    )
    dm = InteractionsDataModule(data_dir=data_dir, batch_size=batch_size)

    logger = MLFlowLogger(
        experiment_name=experiment_path,
        tracking_uri="databricks",
        run_id=run_id,
    )
    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )

    # 1x1에서 strategy="ddp"를 강제하면 단일 프로세스에서 NCCL init이 실패할 수 있다.
    # "auto"는 single process 환경을 안전하게 처리하면서, devices>1 또는 num_nodes>1
    # 일 때만 명시적으로 "ddp"를 켠다.
    strategy = "ddp" if (devices > 1 or num_nodes > 1) else "auto"
    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        max_epochs=num_epochs,
        limit_train_batches=max_steps_per_epoch,
        logger=logger,
        callbacks=[early_stop],
        default_root_dir=ckpt_dir,
    )

    rank = int(os.environ.get("RANK", "0"))
    # rank 0 worker에서만 driver의 run에 attach. 1x1 직접 호출 시에도 RANK 미설정이라 rank=0.
    # multi-node는 driver가 학습에 참여하지 않으므로 worker rank 0에서 attach해야 GPU system metrics가 잡힌다.
    attached = False
    if rank == 0 and mlflow.active_run() is None:
        mlflow.start_run(run_id=run_id, log_system_metrics=True)
        mlflow.log_params(
            {
                "topology": topology,
                "world_size": devices * num_nodes,
                "code_organization": "02-script-based",
            }
        )
        attached = True
    try:
        trainer.fit(model, datamodule=dm)
    finally:
        if attached and mlflow.active_run() is not None:
            mlflow.end_run()
    return "ok"
