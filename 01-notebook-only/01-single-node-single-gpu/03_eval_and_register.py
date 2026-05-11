# Databricks notebook source
# MAGIC %md
# MAGIC # 01-1 · 03_eval_and_register
# MAGIC AUC 평가 → UC Model Registry 등록.

# COMMAND ----------

# TODO: 검증 데이터셋 로드, model.load_state_dict(...)
# TODO: sklearn.metrics.roc_auc_score 계산 + mlflow.log_metric("auc", ...)
# TODO: mlflow.pytorch.log_model(model, "model", registered_model_name=f"{CATALOG}.{SCHEMA}.recommender_1x1")
