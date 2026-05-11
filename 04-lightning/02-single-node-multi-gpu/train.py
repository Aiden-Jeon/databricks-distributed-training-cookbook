# 04-2 · train.py
# Lightning DDP, 1 node × N GPU.

# TODO: trainer = pl.Trainer(devices=4, strategy="ddp", accelerator="gpu", precision="bf16-mixed", logger=MLFlowLogger(...))
# TODO: trainer.fit(RecommenderModule(...), datamodule=RecsysDataModule(...))
