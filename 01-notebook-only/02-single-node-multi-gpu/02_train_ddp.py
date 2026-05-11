# Databricks notebook source
# MAGIC %md
# MAGIC # 01-2 · 02_train_ddp
# MAGIC @distributed(gpus=N) + DDP로 Two-Tower MLP 분산학습.

# COMMAND ----------

# TODO: from databricks.ml.distributed import distributed
# TODO: @distributed(gpus=GPUS_PER_NODE)
# TODO: def train_fn(run_id, data_path, ckpt_dir, n_users, n_items, emb_dim): ...
# TODO:   model = DDP(TwoTowerMLP(...).cuda(), device_ids=[torch.cuda.current_device()])
# TODO:   DistributedSampler(...) + 학습 루프
# TODO: with mlflow.start_run(...) as run: train_fn(run.info.run_id, ...)
