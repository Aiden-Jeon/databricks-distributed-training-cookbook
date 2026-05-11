# Databricks notebook source
# MAGIC %md
# MAGIC # 02-3 · driver_notebook
# MAGIC TorchDistributor.run("train.py", ...) multi-node DDP.

# COMMAND ----------

# TODO: NUM_NODES, GPUS_PER_NODE 변수
# TODO: TorchDistributor(num_processes=M*N, local_mode=False, use_gpu=True).run("train.py", "--config", "...", "--run-id", run.info.run_id)
