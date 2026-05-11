# Databricks notebook source
# MAGIC %md
# MAGIC # Simple Code for Training Huggingface Model

# COMMAND ----------

# MAGIC %pip install -U --quiet ./dist/torch_distributed-0.1.0.tar.gz
# MAGIC %restart_python

# COMMAND ----------
import os
from dataclasses import dataclass


@dataclass
class Arguments:
    log_volume_dir = "/Volumes/jongseob_demo/distributed/pytorch"
    experiment_path = os.path.join(
        os.getcwd().replace("/Workspace", ""), "pytorch_distributor"
    )

    model_name = "distilbert-base-uncased"
    dataset_name = "imdb"

    log_interval = 10
    batch_size = 16
    num_epochs = 1


# COMMAND ----------

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
db_host = context.extraContext().apply("api_url")
db_token = context.apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Train on single node


# COMMAND ----------
def train_single_node(
    model_name,
    dataset_name,
    log_dir,
    experiment_path,
    num_epochs=1,
    batch_size=16,
    log_interval=10,
):
    import mlflow
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from torch_distributed.model import SentimentClassifier
    from torch_distributed.dataset import load_torch_dataset
    from torch_distributed.train import train_one_epoch
    from torch_distributed.utils import save_checkpoint
    from torch_distributed.eval import AverageMeter, evaluate_one_epoch

    #################### Setting up MLflow ####################
    experiment = mlflow.set_experiment(experiment_path)
    ###########################################################

    ###########################################################
    #################### Setting up dataset ###################
    tokenized_datasets = load_torch_dataset(dataset_name, model_name)
    train_dataset = tokenized_datasets["train"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ###########################################################
    ##################### Setting up model ####################
    device = torch.device("cuda")
    model = SentimentClassifier(model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    ###########################################################

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            max_duration=10,
        )
        save_checkpoint(log_dir, model, epoch)

    print("Testing...")
    avg_meter = AverageMeter()
    test_dataset = tokenized_datasets["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    evaluate_one_epoch(
        model=model,
        data_loader=test_dataloader,
        avg_meter=avg_meter,
        log_interval=log_interval,
        max_duration=10,
    )
    test_loss, test_acc = avg_meter.reduce()

    print("Average test loss: {}, accuracy: {}".format(test_loss, test_acc))
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)
    return "Finished"


# COMMAND ----------
import mlflow
from torch_distributed.utils import create_log_dir

args = Arguments()
log_dir = create_log_dir(args.log_volume_dir)
print("Log directory:", log_dir)
with mlflow.start_run():
    train_single_node(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        log_dir=log_dir,
        experiment_path=args.experiment_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )
