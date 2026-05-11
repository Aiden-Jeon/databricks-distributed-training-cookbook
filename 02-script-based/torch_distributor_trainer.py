"""TorchDistributor / HF Accelerate가 entry point로 사용하는 DDP 학습 함수.

- TorchDistributor: `TorchDistributor(...).run(train_fn, **kwargs)` 형태로 호출.
- accelerate launch: `accelerate launch torch_distributor_trainer.py --args ...` 형태로 실행.
  Accelerate launcher도 TorchDistributor와 동일하게 RANK / WORLD_SIZE / LOCAL_RANK /
  MASTER_ADDR / MASTER_PORT 환경변수를 세팅하므로, 함수 본문은 두 경우 모두에서 그대로 동작한다.

토폴로지(1x1 / 1xN / MxN)는 launcher 측 인자(num_processes, local_mode 또는 --num_processes)로
전환되며 본 모듈 내부 분기는 없다. multi-node에서 worker 프로세스가 sibling .py를 import할 수
있도록 호출 측이 script_dir을 인자로 넘겨주면, child 함수가 sys.path에 삽입한다.
"""

import argparse
import os
import sys


def train_fn(
    run_id,
    db_host,
    db_token,
    data_dir,
    ckpt_path,
    n_users,
    n_items,
    emb_dim,
    tower_hidden,
    batch_size,
    num_epochs,
    max_steps_per_epoch,
    patience,
    min_delta,
    topology,
    script_dir,
):
    """DDP 학습 entrypoint. 1x1 / 1xN / MxN 모두 동일 시그니처."""
    # multi-node 워커 프로세스는 fresh Python으로 시작하므로 sibling 모듈 import를 위해
    # script_dir을 sys.path에 직접 추가한다. single-node에서도 무해.
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import mlflow
    import pyarrow.parquet as pq
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    from model import EarlyStopping, TwoTowerMLP

    # MLflow tracking이 child에서 동일 워크스페이스를 인증할 수 있도록 driver의 자격증명 전달.
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token

    # `accelerate launch --num_processes 1`은 simple_launcher 경로라 RANK/WORLD_SIZE 등을
    # 세팅하지 않는다. TorchDistributor 및 `accelerate launch --multi_gpu`는 이미 세팅된
    # 값을 그대로 두고, 미설정 시에만 단일 프로세스 기본값을 채워 init_process_group이 환경
    # rendezvous에서 ValueError를 던지지 않도록 한다.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    def load_split(split):
        split_dir = os.path.join(data_dir, split)
        files = sorted(
            os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".parquet")
        )
        table = pq.read_table(files, columns=["user_id", "item_id", "label"])
        return TensorDataset(
            torch.from_numpy(table.column("user_id").to_numpy()),
            torch.from_numpy(table.column("item_id").to_numpy()),
            torch.from_numpy(table.column("label").to_numpy()),
        )

    train_dataset = load_split("train")
    val_dataset = load_split("val")
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=global_rank, drop_last=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=True
    )

    model = TwoTowerMLP(n_users, n_items, emb_dim, tower_hidden).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta)

    if global_rank == 0:
        mlflow.start_run(run_id=run_id, log_system_metrics=True)
        mlflow.log_params({
            "topology": topology,
            "world_size": world_size,
            "n_users": n_users,
            "n_items": n_items,
            "emb_dim": emb_dim,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "max_steps_per_epoch": max_steps_per_epoch,
            "patience": patience,
            "min_delta": min_delta,
            "code_organization": "02-script-based",
        })

    global_step = 0
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for step, (u, i, y) in enumerate(train_loader):
            if step >= max_steps_per_epoch:
                break
            u, i, y = u.to(device), i.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(u, i)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            if global_rank == 0 and step % 20 == 0:
                mlflow.log_metric("train/loss", loss.item(), step=global_step)
            global_step += 1

        model.eval()
        local_loss_sum = torch.zeros(1, device=device)
        local_n = torch.zeros(1, device=device)
        with torch.no_grad():
            for u, i, y in val_loader:
                u, i, y = u.to(device), i.to(device), y.to(device)
                logits = model(u, i)
                loss = loss_fn(logits, y)
                local_loss_sum += loss * y.size(0)
                local_n += y.size(0)
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_n, op=dist.ReduceOp.SUM)
        val_loss = (local_loss_sum / local_n).item()

        if global_rank == 0:
            mlflow.log_metric("val/loss", val_loss, step=epoch)
            print(
                f"[rank=0] epoch={epoch:2d} val_loss={val_loss:.6f} "
                f"(best={early_stop.best:.6f} counter={early_stop.counter})"
            )

        if early_stop.step(val_loss):
            if global_rank == 0:
                print(f"early stop at epoch {epoch} (best val_loss={early_stop.best:.6f})")
                mlflow.log_metric("early_stop_epoch", epoch)
            break

    dist.barrier()
    if global_rank == 0:
        torch.save({"model": model.module.state_dict()}, ckpt_path)
        mlflow.log_param("ckpt_path", ckpt_path)
        mlflow.end_run()
        print(f"saved {ckpt_path}")
    dist.destroy_process_group()
    return "ok"


def _parse_args():
    """accelerate launch entry point용 argparse."""
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--db_host", required=True)
    p.add_argument("--db_token", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--n_users", type=int, required=True)
    p.add_argument("--n_items", type=int, required=True)
    p.add_argument("--emb_dim", type=int, required=True)
    p.add_argument(
        "--tower_hidden",
        type=int,
        nargs="+",
        required=True,
        help="space-separated ints, e.g. --tower_hidden 256 128",
    )
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--num_epochs", type=int, required=True)
    p.add_argument("--max_steps_per_epoch", type=int, required=True)
    p.add_argument("--patience", type=int, required=True)
    p.add_argument("--min_delta", type=float, required=True)
    p.add_argument("--topology", required=True)
    p.add_argument(
        "--script_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="sibling .py 모듈(model.py 등) import를 위한 디렉토리",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_fn(
        run_id=args.run_id,
        db_host=args.db_host,
        db_token=args.db_token,
        data_dir=args.data_dir,
        ckpt_path=args.ckpt_path,
        n_users=args.n_users,
        n_items=args.n_items,
        emb_dim=args.emb_dim,
        tower_hidden=tuple(args.tower_hidden),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        patience=args.patience,
        min_delta=args.min_delta,
        topology=args.topology,
        script_dir=args.script_dir,
    )
