"""TorchDistributor / HF Accelerate가 entry point로 사용하는 DDP 학습 함수.

- TorchDistributor: `TorchDistributor(...).run(train_fn, **kwargs)` 형태로 호출.
- accelerate launch: notebook-scoped Python env(`sys.executable`)에서 호출해야 한다.
  시스템의 `/databricks/python3/bin/accelerate`는 `recommender_pkg`이 설치된 env를 보지
  못한다. 호출 형식:
      sys.executable -m accelerate.commands.accelerate_cli launch \
          -m recommender_pkg.torch_distributor_trainer --args ...
  `-m` 모듈 모드를 써야 sys.path[0]이 패키지 내부로 잡히는 문제를 피한다. Accelerate
  launcher도 TorchDistributor와 동일하게 RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR /
  MASTER_PORT 환경변수를 세팅하므로, 함수 본문은 두 경우 모두에서 그대로 동작한다.

02-script-based/torch_distributor_trainer.py와 함수 시그니처·동작은 동일하며, 차이는
import 경로(`recommender_pkg.model`)뿐. 패키지가 wheel로 설치되어 있으므로 `script_dir`
인자는 호환성을 위해 유지하지만 sys.path 보강이 불필요하다.

토폴로지(1x1 / 1xN / MxN)는 launcher 측 인자(num_processes, local_mode 또는 --num_processes)로
전환되며 본 모듈 내부 분기는 없다.
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
    # 02-script-based와의 시그니처 동등성을 위해 script_dir을 유지. 패키지가 설치되어
    # 있으므로 일반적으로 sys.path 보강은 불필요하지만, 호출 측이 값을 넘기면 무해하게
    # 추가한다.
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

    from recommender_pkg.model import EarlyStopping, TwoTowerMLP

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
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".parquet")
        )
        table = pq.read_table(files, columns=["user_id", "item_id", "label"])
        return TensorDataset(
            torch.from_numpy(table.column("user_id").to_numpy()),
            torch.from_numpy(table.column("item_id").to_numpy()),
            torch.from_numpy(table.column("label").to_numpy()),
        )

    train_dataset = load_split("train")
    val_dataset = load_split("val")
    # train: drop_last=True — 각 rank의 step 수를 동일하게 맞춰 누가 먼저 끝나면서
    # AllReduce를 기다리는 hang을 막는다 (common-pitfalls §4).
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=global_rank, drop_last=True
    )
    # val: shuffle=False (deterministic 평가), drop_last=False (전체 데이터 사용).
    # 마지막 batch가 rank마다 크기가 다를 수 있으나, 아래 loss/n을 all_reduce(SUM) 후
    # 나누는 방식이라 unbiased로 평균이 나온다.
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
    )

    model = TwoTowerMLP(n_users, n_items, emb_dim, tower_hidden).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta)

    if global_rank == 0:
        mlflow.start_run(run_id=run_id, log_system_metrics=True)
        mlflow.log_params(
            {
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
                "code_organization": "03-custom-package-script-based",
            }
        )

    global_step = 0
    for epoch in range(num_epochs):
        # DistributedSampler가 epoch마다 다른 shuffle seed를 쓰도록 알려준다.
        # 빼먹으면 모든 epoch이 동일한 순서로 돌아 학습 dynamics가 망가진다.
        train_sampler.set_epoch(epoch)
        model.train()
        for step, (u, i, y) in enumerate(train_loader):
            if step >= max_steps_per_epoch:
                break
            u, i, y = u.to(device), i.to(device), y.to(device)
            # set_to_none=True: grad를 0 tensor로 채우지 않고 None으로 해제. 다음
            # backward에서 새로 할당하므로 메모리 회수 효과. 임베딩처럼 큰 grad 텐서에서
            # 유의미한 절감.
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

        # 모든 rank가 all_reduce된 동일한 val_loss로 step() 호출 → 같은 시점에 break.
        # 이게 보장돼야 rank 간 학습 종료 시점이 어긋나 hang 되지 않는다 (model.py 의
        # EarlyStopping docstring 참고).
        if early_stop.step(val_loss):
            if global_rank == 0:
                print(
                    f"early stop at epoch {epoch} (best val_loss={early_stop.best:.6f})"
                )
                mlflow.log_metric("early_stop_epoch", epoch)
            break

    # rank 0가 ckpt 저장하는 동안 다른 rank가 먼저 destroy_process_group으로 들어가
    # NCCL 통신이 깨지지 않도록 동기화.
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
        default="",
        help="02-script-based와의 호환을 위한 인자. 패키지가 설치되어 있으면 비워둔다.",
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
