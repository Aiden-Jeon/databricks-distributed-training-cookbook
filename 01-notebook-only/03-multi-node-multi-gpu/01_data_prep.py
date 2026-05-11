# Databricks notebook source
# MAGIC %md
# MAGIC # 01-3 · 01_data_prep
# MAGIC 합성 interaction → parquet shard (UC Volume).

# COMMAND ----------

# TODO: rng로 (user_id, item_id, label) 생성, pyarrow.parquet으로 shard별 저장
# TODO: 출력 경로 /Volumes/<catalog>/<schema>/interactions_large/part-*.parquet
