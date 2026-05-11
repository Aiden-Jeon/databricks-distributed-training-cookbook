import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchmetrics import Accuracy
from composer.models import ComposerModel


class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


class ComposerSentimentClassifier(ComposerModel):
    def __init__(self, model_name, num_labels=2):
        super(ComposerSentimentClassifier, self).__init__()
        self.model = SentimentClassifier(model_name, num_labels)
        self.train_accuracy = Accuracy(task="binary", average="micro")
        self.val_accuracy = Accuracy(task="binary", average="micro")

    def forward(self, batch):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        return self.model(input_ids, attention_mask)

    def loss(self, outputs, batch):
        labels = batch["label"]
        return F.cross_entropy(outputs, labels)

    def eval_forward(self, batch, outputs=None):
        if outputs is not None:
            return outputs.argmax(dim=1)
        outputs = self.forward(batch)
        return outputs.argmax(dim=1)

    def update_metric(self, batch, outputs, metric):
        labels = batch["label"]
        metric.update(outputs, labels)

    def get_metrics(self, is_train=False):
        # defines which metrics to use in each phase of training
        device = next(self.parameters()).device
        return (
            {"Accuracy": self.train_accuracy.to(device)}
            if is_train
            else {"Accuracy": self.val_accuracy.to(device)}
        )


class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def predict(self, context, model_input, params=None):
        input_ids, attention_mask = (
            model_input["input_ids"],
            model_input["attention_mask"],
        )
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)
        output = self.model(input_ids, attention_mask)
        return output.cpu().numpy()
