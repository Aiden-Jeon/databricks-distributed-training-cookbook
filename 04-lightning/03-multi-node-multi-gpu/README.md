# 04-3 · Lightning · M×N GPU

> `Trainer(num_nodes=M, devices=N, strategy="ddp")` + TorchDistributor wrap으로 multi-node DDP 학습. 본 쿡북 Lightning 트랙의 최종 형태.

## 🎯 시나리오

- 04-2의 LightningModule을 그대로 multi-node로 확장
- Lightning을 끝까지 유지하고 노드만 늘리는 운영 패턴

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (user 10M × item 5M, emb dim 256) |
| 라이브러리 | `pytorch-lightning`, `torch` |
| 병렬화 | **DDP** (multi-node) |
| 데이터 | parquet shard (UC Volume) |
| 실행 | TorchDistributor가 각 노드에서 `train.py`를 실행, 내부에서 Lightning이 `num_nodes`/`devices`로 rendezvous |
| 추적 | `MLFlowLogger` (rank 0) |

## 🖥️ 클러스터 권장 사양

[01-3과 동일](../../01-notebook-only/03-multi-node-multi-gpu/README.md#️-클러스터-권장-사양). Classic 멀티 노드.

## 📂 파일

```
03-multi-node-multi-gpu/
├── README.md
├── driver_notebook.py
└── train.py
```

## 🚀 실행 순서

1. driver에서 `NUM_NODES`·`GPUS_PER_NODE` 변수 설정.
2. `TorchDistributor.run("train.py", "--num-nodes", str(M), "--devices", str(N), ...)`
3. driver에서 등록.

## 🧬 핵심 패턴

```python
# train.py (entrypoint)
import argparse
import lightning.pytorch as pl

parser = argparse.ArgumentParser()
parser.add_argument("--num-nodes", type=int, required=True)
parser.add_argument("--devices", type=int, required=True)
args = parser.parse_args()

trainer = pl.Trainer(
    num_nodes=args.num_nodes,
    devices=args.devices,
    accelerator="gpu",
    strategy="ddp",
    precision="bf16-mixed",
)
trainer.fit(module, datamodule=dm)
```

```python
# driver_notebook.py
TorchDistributor(
    num_processes=NUM_NODES * GPUS_PER_NODE,
    local_mode=False,
    use_gpu=True,
).run("train.py", "--num-nodes", str(NUM_NODES), "--devices", str(GPUS_PER_NODE))
```

## ⚠️ 함정

- Lightning의 `num_nodes`와 TorchDistributor의 `num_processes`가 일관되게 곱셈 관계여야 한다.
- 노트북 셀의 stdout이 너무 길어지면 Lightning progress bar를 끄는 게 안전 (`enable_progress_bar=False`).
- 작은 모델 + 다수 노드 조합에서 throughput 한계 ([common-pitfalls #9](../../00-foundations/common-pitfalls.md#9-multi-node-ddp인데-throughput이-안-올라간다)).

## ➡️ 다음 셀

- 옆: 매트릭스의 마지막 셀.
- 위: [04-2](../02-single-node-multi-gpu/), [03-3](../../03-cli-accelerate/03-multi-node-multi-gpu/), [02-3](../../02-script-based/03-multi-node-multi-gpu/)
- 아래: 끝.

## 📚 출처/참고

- Lightning multi-node: https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_1.html
- TorchDistributor: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor
