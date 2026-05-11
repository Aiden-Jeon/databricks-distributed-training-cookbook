# 02-3 · train.py
# Multi-node DDP entrypoint. argparse로 cfg + run_id 받음.

# TODO: argparse: --config, --run-id
# TODO: DDP(TwoTowerMLP(...).cuda()) + DistributedSampler(parquet dataset)
# TODO: 종료 시 /local_disk0 → /Volumes copy (rank 0)
