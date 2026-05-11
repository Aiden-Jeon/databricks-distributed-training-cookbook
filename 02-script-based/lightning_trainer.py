"""PyTorch Lightning 기반 학습.

03-launch_lightning_trainer.ipynb가 이 모듈을 import해서 Trainer를 호출한다.
토폴로지는 Trainer(devices=N, num_nodes=M, strategy="ddp")로 결정.
"""

# TODO: import pytorch_lightning as pl
# TODO: from model import TwoTowerMLP
# TODO: class RecommenderLightning(pl.LightningModule):
# TODO:   - __init__: TwoTowerMLP wrap
# TODO:   - training_step / validation_step (sync_dist=True)
# TODO:   - configure_optimizers

# TODO: def fit(data_path, ckpt_dir, devices, num_nodes,
# TODO:         n_users, n_items, emb_dim, batch_size, num_epochs):
# TODO:   - MLFlowLogger 구성
# TODO:   - Trainer(accelerator="gpu", devices=devices, num_nodes=num_nodes,
# TODO:             strategy="ddp", logger=logger, ...)
# TODO:   - trainer.fit(model, datamodule)
