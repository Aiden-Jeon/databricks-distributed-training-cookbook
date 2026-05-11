# 02-2 · train.py
# Two-Tower MLP, DDP entrypoint.

# TODO: def main(cfg_path, run_id):
# TODO:   model = DDP(TwoTowerMLP(...).cuda(), device_ids=[torch.cuda.current_device()])
# TODO:   DistributedSampler(...) + 학습 루프
