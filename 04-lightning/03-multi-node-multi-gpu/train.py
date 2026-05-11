# 04-3 · train.py
# Lightning multi-node DDP entrypoint.

# TODO: argparse: --num-nodes, --devices
# TODO: pl.Trainer(num_nodes=M, devices=N, strategy="ddp", accelerator="gpu", precision="bf16-mixed")
# TODO: trainer.fit(RecommenderModule(...), datamodule=RecsysDataModule(...))
