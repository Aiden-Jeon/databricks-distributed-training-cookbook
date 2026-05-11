# Databricks notebook source
# MAGIC %md
# MAGIC # 02-2 · driver_notebook
# MAGIC @distributed(gpus=N) 내부에서 train.main(cfg) 호출.

# COMMAND ----------

# TODO: @distributed(gpus=N) def launch(cfg_path, run_id): import train; train.main(cfg_path, run_id=run_id)
# TODO: with mlflow.start_run() as run: launch("configs/training_args.yaml", run.info.run_id)
