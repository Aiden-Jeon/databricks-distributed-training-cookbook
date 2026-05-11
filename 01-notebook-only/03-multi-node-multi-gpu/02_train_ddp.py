# Databricks notebook source
# MAGIC %md
# MAGIC # 01-3 · 02_train_ddp
# MAGIC TorchDistributor(local_mode=False).run(train_fn, ...) multi-node DDP.

# COMMAND ----------

# TODO: def train_fn(run_id, data_path, ckpt_dir, n_users, n_items, emb_dim):
# TODO:   DDP(TwoTowerMLP(...).cuda()) + DistributedSampler(parquet dataset)
# TODO: TorchDistributor(num_processes=NUM_NODES*GPUS_PER_NODE, local_mode=False, use_gpu=True).run(train_fn, ...)
