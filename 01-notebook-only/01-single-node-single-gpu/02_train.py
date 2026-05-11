# Databricks notebook source
# MAGIC %md
# MAGIC # 01-1 · 02_train
# MAGIC Two-Tower MLP, 단일 GPU 학습 루프.

# COMMAND ----------

# TODO: mlflow.pytorch.autolog()
# TODO: model = TwoTowerMLP(N_USERS, N_ITEMS, EMB_DIM).to("cuda")
# TODO: loader = DataLoader(TensorDataset(...), batch_size=4096, shuffle=True)
# TODO: with mlflow.start_run(run_name="recommender-1x1"): 학습 루프 → torch.save(state_dict)
