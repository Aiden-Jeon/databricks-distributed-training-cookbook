# 03-2 · train.py
# Two-Tower MLP, accelerate launch entrypoint, 1×N GPU DDP.

# TODO: accelerator = Accelerator(mixed_precision="bf16")
# TODO: model, optimizer, loader = accelerator.prepare(TwoTowerMLP(...), AdamW(...), DataLoader(...))
# TODO: 학습 루프 (accelerator.backward(loss))
