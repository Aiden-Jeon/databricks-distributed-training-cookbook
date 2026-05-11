# 04-1 · train.py
# Two-Tower MLP + LightningModule + DataModule + Trainer(devices=1).

# TODO: class RecommenderModule(pl.LightningModule): training_step / configure_optimizers
# TODO: class RecsysDataModule(pl.LightningDataModule): train_dataloader
# TODO: trainer = pl.Trainer(devices=1, accelerator="gpu", precision="bf16-mixed", logger=MLFlowLogger(...))
# TODO: trainer.fit(RecommenderModule(...), datamodule=RecsysDataModule(...))
