# Databricks notebook source
# MAGIC %md
# MAGIC # 04-3 · driver_notebook
# MAGIC TorchDistributor.run("train.py", "--num-nodes", M, "--devices", N).

# COMMAND ----------

# TODO: TorchDistributor(num_processes=M*N, local_mode=False, use_gpu=True).run("train.py", "--num-nodes", str(M), "--devices", str(N))
