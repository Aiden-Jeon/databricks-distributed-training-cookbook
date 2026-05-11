# Databricks notebook source
# MAGIC %md
# MAGIC # 03-3 · driver_notebook
# MAGIC TorchDistributor가 각 노드에서 accelerate launch (DDP)를 호출.

# COMMAND ----------

# TODO: def launch_accelerate(cfg_yaml, train_args):
# TODO:     subprocess.check_call(["accelerate", "launch", "--config_file", cfg_yaml, "train.py", *train_args])
# TODO: TorchDistributor(num_processes=NUM_NODES, local_mode=False, use_gpu=True).run(launch_accelerate, "configs/accelerate_ddp_multinode.yaml", ...)
