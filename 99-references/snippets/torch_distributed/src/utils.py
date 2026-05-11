import os
from time import time

import torch


def create_log_dir(log_volume_dir):
    log_dir = os.path.join(log_volume_dir, str(time()))
    os.makedirs(log_dir)
    return log_dir


def create_run_name(run_name):
    return f"{run_name}-{str(time())}"


def save_checkpoint(log_dir, model, epoch):
    filepath = log_dir + "/checkpoint-{epoch}.pth.tar".format(epoch=epoch)
    state = {
        "model": model.state_dict(),
    }
    torch.save(state, filepath)
