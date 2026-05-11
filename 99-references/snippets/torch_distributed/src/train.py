import mlflow
import torch.nn.functional as F


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    log_interval,
    max_duration=None,
):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        total_loss += loss

        mlflow.log_metric("train_loss", loss)
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(input_ids),
                    len(data_loader) * len(input_ids),
                    100.0 * batch_idx / len(data_loader),
                    loss,
                )
            )
        if max_duration is not None:
            if batch_idx == max_duration:
                break
    return total_loss / len(data_loader)
