"""TorchDistributor / HF Accelerate가 entry point로 사용하는 학습 함수.

- 02-launch_torch_distributor.ipynb: TorchDistributor.run(train_fn, ...)
- 04-launch_accelerator.ipynb: accelerate launch torch_distributor_trainer.py

토폴로지(1x1 / 1xN / MxN)는 launcher의 인자로 전환된다. 본 모듈 내부는 동일.
"""

# TODO: import torch, torch.distributed, mlflow
# TODO: from model import TwoTowerMLP
# TODO: def train_fn(run_id, db_host, db_token, data_path, ckpt_dir,
# TODO:              n_users, n_items, emb_dim, batch_size, num_epochs):
# TODO:   - os.environ DATABRICKS_HOST/TOKEN 세팅
# TODO:   - dist.init_process_group("nccl")
# TODO:   - DDP wrap + DistributedSampler
# TODO:   - rank 0에서만 mlflow.start_run(run_id=run_id)
# TODO:   - 학습 루프
# TODO:   - rank 0에서만 save_checkpoint
# TODO:   - dist.destroy_process_group()

# TODO: if __name__ == "__main__":  # accelerate launch 진입점
# TODO:     # argparse로 인자 받고 train_fn 호출
