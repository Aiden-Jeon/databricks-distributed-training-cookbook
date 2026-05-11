import torch
import torch.nn.functional as F
import torch.distributed as dist


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0
        self.correct = 0
        self.total = 0

    def update(self, loss, correct, total):
        self.loss = loss
        self.correct = correct
        self.total = total

    def all_reduce(self, device):
        total = torch.tensor([self.loss, self.correct, self.total], device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        loss, correct, total = total.tolist()
        test_loss = loss / total
        test_acc = correct / total
        return test_loss, test_acc

    def reduce(self):
        test_loss = self.loss / self.total
        test_acc = self.correct / self.total
        return test_loss, test_acc


@torch.no_grad()
def evaluate_one_epoch(
    model,
    data_loader,
    avg_meter,
    log_interval,
    max_duration=None,
):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(data_loader):
        device = torch.device("cuda")
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["label"].to(device),
        )
        outputs = model(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels).item()

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        avg_meter.update(loss, correct, total)
        if batch_idx % log_interval == 0:
            print(
                "Test [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(input_ids),
                    len(data_loader) * len(input_ids),
                    100.0 * batch_idx / len(data_loader),
                    loss,
                )
            )
        if max_duration is not None:
            if batch_idx == max_duration:
                break
    return loss, correct, total
