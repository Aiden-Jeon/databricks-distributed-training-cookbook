# Distributed Training Concepts

이 문서는 쿡북 전반에서 반복해서 등장하는 핵심 개념을 한자리에 모아 둔 글입니다. 다른 곳에서 같은 설명을 반복하지 않기 위한 공용 어휘집이라고 보면 됩니다.

## 본 쿡북이 다루는 병렬화: DDP 한 가지

분산 학습 전략은 여러 가지가 있지만, 본 쿡북은 **DDP(DistributedDataParallel) 한 가지만** 다룹니다. 각 GPU에 모델을 통째로 복제해 두고 데이터를 나눠 처리하는 가장 기본적인 데이터 병렬 방식입니다.

| 항목 | DDP (DistributedDataParallel) |
|------|-------------------------------|
| 모델 메모리 | 각 GPU에 모델 **전체** 복제 |
| 그래디언트 | 각 GPU 계산 후 AllReduce |
| 데이터 | rank별로 미니배치를 나눠 처리 |
| 통신량 | 작음 (AllReduce 1회/스텝) |
| 적합 모델 크기 | 한 GPU에 모델 + optimizer state가 들어가는 크기 |

본 쿡북의 기준 모델([concepts-recommender-baseline.md](concepts-recommender-baseline.md))은 가장 큰 설정에서도 수 GB 규모의 임베딩 테이블과 작은 MLP로 구성되어 단일 GPU에 무리 없이 들어갑니다. 모델을 GPU 사이에 쪼개야 할 만큼 크지 않다는 뜻이며, 따라서 **DDP만으로 충분합니다**. 모델 자체를 샤딩해야 하는 FSDP는 이 쿡북의 범위 밖이라 다루지 않습니다.

> 토폴로지(1×1 / 1×N / M×N)와 DDP를 혼동하기 쉽지만 둘은 층위가 다릅니다. 토폴로지는 **GPU를 몇 개 어떻게 묶을지**의 문제이고, DDP는 **그 GPU들에 같은 모델을 복제해 데이터를 병렬 처리하는 전략**입니다. 1×N이든 M×N이든 학습 전략은 모두 DDP이며, 차이는 GPU가 한 노드 안에 모여 있느냐, 노드를 가로지르느냐뿐입니다.

## TorchDistributor `local_mode`: driver vs worker

TorchDistributor에서 가장 먼저 부딪히는 설정이 `local_mode` 플래그입니다. 이름이 다소 모호하지만 실제로는 **분산 학습 프로세스를 어디에 띄울 것인가** — driver 노드 안인지, Spark worker 노드들에 분산할 것인지를 결정합니다.

| `local_mode` | child 프로세스가 도는 곳 | 사용 토폴로지 | driver의 역할 |
|--------------|------------------------|--------------|--------------|
| `True` | **driver 노드 안에서** N개 프로세스 spawn | 1×1, 1×N (single node) | driver가 곧 학습 노드. driver의 GPU를 그대로 사용. |
| `False` | Spark **worker 노드들**에 분산 spawn | M×N (multi node) | driver는 코디네이션만. 학습에는 참여하지 않음. |

표를 풀어 쓰면 다음과 같습니다.

- `local_mode=True`는 "driver에서 학습한다"는 뜻으로 읽으면 됩니다. Single-node 클러스터(driver만 있고 worker=0)나 driver에 GPU가 붙어 있는 클러스터에서 사용합니다. TorchDistributor가 driver 머신 안에서 `num_processes`개의 Python 프로세스를 띄우고 NCCL로 묶어 줍니다.
- `local_mode=False`는 반대로 "worker에서 학습한다"는 뜻입니다. Spark가 worker 노드들에 child 프로세스를 분배하고, driver는 학습 자체에는 끼지 않고 RPC 코디네이션만 담당합니다. 이 차이는 모니터링에도 그대로 영향을 주는데, multi-node에서 driver-side `log_system_metrics=True`만 켜면 driver는 학습을 하지 않기 때문에 idle 메트릭만 기록됩니다 ([`ops-mlflow-tracking.md §1`](ops-mlflow-tracking.md)).

본 쿡북의 각 토폴로지가 위 두 모드 중 무엇에 해당하는지 정리하면 다음과 같습니다.

| 셀 | 호출 | 의미 |
|----|------|------|
| 1×1 | `TorchDistributor(num_processes=1, local_mode=True)` | driver에서 1개 프로세스. world_size=1이라 DDP all_reduce는 no-op. |
| 1×N | `TorchDistributor(num_processes=N, local_mode=True)` | driver의 N개 GPU에 N개 프로세스. |
| M×N | `TorchDistributor(num_processes=M*N, local_mode=False)` | M개 worker 노드에 총 M*N 프로세스 분배. driver는 코디네이션만. |

이 플래그를 잘못 맞추면 학습 자체가 시작되지 않거나, 시작되더라도 의도와 전혀 다른 모양으로 돌아갑니다. 자주 마주치는 함정은 다음과 같습니다.

- Single-node 클러스터에서 `local_mode=False`로 호출하면 분배할 worker가 없어 그대로 실패합니다.
- Multi-node 클러스터에서 `local_mode=True`로 호출하면 비싼 worker 노드들이 놀고 driver 한 대에서만 학습이 진행됩니다.
- `local_mode=False`로 띄운 child 프로세스는 다른 노드의 fresh Python에서 시작하므로 driver의 자격증명이나 import 상태를 자동으로 상속받지 못합니다. 실수의 단골 원인이니 따로 살펴 두는 것이 좋습니다 ([`debug-common-pitfalls.md §2`, `§2-1`](debug-common-pitfalls.md)).

## Lightning에는 왜 `local_mode`가 없는가 — driver는 어떻게 쓰이는가

`local_mode`는 TorchDistributor의 **launcher 옵션**입니다. PyTorch Lightning은 launcher가 아니라 training framework이고, 분산 환경 자체는 외부 launcher(`torchrun`, SLURM, TorchDistributor)가 잡아준다는 전제 위에서 동작합니다.

> "Lightning implementation of DDP calls your script under the hood multiple times with the correct environment variables." / "simply launch your script with the torchrun command." — [PyTorch Lightning Docs: GPU training (Intermediate)](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html)

그래서 `L.Trainer`는 `devices`, `num_nodes`, `strategy`만 받습니다. "어디(driver/worker)에서 프로세스를 띄울지"는 Trainer가 모르고, 그건 위에서 누가 띄우느냐 — 즉 launcher — 에 달려 있습니다.

Driver 사용 여부는 결국 **launcher가 결정**합니다:

| 시나리오 | 프로세스를 띄우는 주체 | driver 사용 |
|---------|--------------------|------------|
| 노트북에서 `Trainer.fit()` 직접 호출 (single-node) | Lightning이 노트북 프로세스 자체에서 child 1~N개 spawn | O — driver의 GPU 사용 |
| `TorchDistributor(local_mode=True)` + Lightning | TorchDistributor가 driver 안에서 N개 프로세스 spawn, 각자가 `Trainer.fit()` 호출 | O |
| `TorchDistributor(local_mode=False)` + Lightning | Spark가 worker 노드들에 분산 spawn | X — driver는 코디네이션만 |
| 노트북에서 `Trainer(num_nodes=M>1)` 직접 호출 | Lightning은 worker 노드에 프로세스를 띄울 수단이 없음 | **실패** |

본 쿡북의 매핑 (`01-notebook-based/04-train_pytorch_lightning_1x1_1xN.ipynb`, `05-train_pytorch_lightning_MxN.ipynb`):

| 셀 | launcher | Trainer 호출 |
|----|----------|-------------|
| Lightning 1×1 | 없음 — 노트북에서 직접 | `Trainer(devices=1, num_nodes=1).fit(...)` |
| Lightning 1×N | `TorchDistributor(num_processes=N, local_mode=True)` | `Trainer(devices=N, num_nodes=1, strategy="ddp").fit(...)` |
| Lightning M×N | `TorchDistributor(num_processes=M*N, local_mode=False)` | `Trainer(devices=N, num_nodes=M, strategy="ddp").fit(...)` |

1×1을 TorchDistributor 없이 직접 호출하는 이유: Lightning 단독으로 single-process 학습이 완결되며, launcher를 한 겹 더 두는 가치가 거의 없습니다. 반면 1×N부터는 TorchDistributor가 process 스폰과 NCCL 환경변수(`RANK`, `WORLD_SIZE`, `MASTER_ADDR` 등)를 일관되게 잡아주므로 감쌉니다. M×N은 worker 노드에 프로세스를 분배해 줄 주체가 필요하므로 **반드시** `local_mode=False`로 감쌉니다 ([Databricks Blog: PyTorch on Databricks – Introducing the Spark PyTorch Distributor](https://www.databricks.com/blog/2023/04/20/pytorch-databricks-introducing-spark-pytorch-distributor.html)).

> Databricks 공식 블로그는 multi-node Lightning에서 `Trainer(devices=1, num_nodes=num_processes)`도 권하는 패턴으로 소개합니다. 이는 각 TorchDistributor 프로세스를 독립 "node"로 취급해 Lightning이 추가 child를 spawn하지 않게 만드는 단순화 패턴입니다. 본 쿡북은 클러스터 토폴로지를 그대로 드러내기 위해 `devices=N, num_nodes=M`을 사용합니다. 둘 다 작동하지만 의미와 디버깅 관점이 다릅니다.

함정
- 노트북에서 `Trainer(num_nodes=M>1)`를 TorchDistributor 없이 직접 호출하면 Lightning이 worker 노드에 접근할 수단이 없어 멈춥니다. Databricks에서 multi-node Lightning은 반드시 `TorchDistributor(local_mode=False)`로 감쌉니다.
- `TorchDistributor(local_mode=True, num_processes=N)`로 감싼 안쪽에서 `Trainer(num_nodes>1)`를 주면 안 됩니다. TorchDistributor가 driver 한 대 안에서 N개 프로세스를 띄우는데, Lightning 입장에서는 "여러 노드"라 거짓말을 하는 셈이라 rendezvous가 어긋납니다.
- TorchDistributor child는 노트북과 별개의 fresh Python으로 시작합니다. LightningModule/DataModule, callback import는 모두 `train_fn` 안에서 다시 정의·import 해야 합니다 ([`debug-common-pitfalls.md §2`](debug-common-pitfalls.md)).

## 왜 multi-node DDP인가 (M×N 셀의 정당화)

작은 MLP에서 multi-node DDP는 두 가지 이유로 의미가 있습니다:

1. **데이터 처리량 확장.** 수십억 행 user-item interaction을 노드 수에 비례하게 처리할 수 있습니다.
2. **TorchDistributor 사용법 시연.** Databricks에서 multi-node 분산 launcher를 어떻게 띄우는지를 같은 모델 골격으로 보여주기 위함.

> 다만 모델이 작을수록 AllReduce 통신 비용이 상대적으로 커집니다. 노드를 무작정 늘리면 throughput이 비례하지 않을 수 있습니다. 본 쿡북의 M×N 셀들은 "패턴이 동작합니다"는 것을 보이는 데 초점을 두며, 본격 스케일링 실험은 사용자 워크로드에 맞춰 별도로 진행합니다.

## Mixed precision

| 모드 | 설명 | 사용 시점 |
|------|------|----------|
| FP32 | 기준점, 메모리 4×param | 디버깅용 |
| FP16 + GradScaler | 절반 메모리, overflow 위험 | 구형 GPU(V100) |
| BF16 | 절반 메모리, scaler 불필요 | Ampere(A100)/Hopper(H100). **기본 선택**. |

본 쿡북의 셀은 모두 BF16을 가정합니다. 모델이 작아 mixed precision 효과가 LLM만큼 극적이진 않지만, AdamW 옵티마이저 상태(파라미터의 2배)까지 고려하면 메모리 절감은 여전히 의미가 있습니다.

## 임베딩 테이블이 큰 경우

추천 모델에서 파라미터 대부분은 `nn.Embedding`에서 나옵니다. DDP는 임베딩 테이블도 GPU마다 복제하므로 다음을 염두에 둡니다:

- 임베딩 테이블이 GPU 메모리의 큰 비중을 차지하면 `per_device_batch_size`를 충분히 줄여야 합니다.
- `optimizer.zero_grad(set_to_none=True)`로 그래디언트 메모리를 회수합니다.
- 정말 거대한 임베딩(수십~수백 GB)이라면 `TorchRec`처럼 임베딩을 GPU 간 샤딩하는 라이브러리가 필요하지만 본 쿡북 범위 밖입니다.

## 학습률·배치·grad accumulation

- 분산 학습은 **effective batch size = per_device_batch × num_gpus × grad_accum_steps**.
- 학습률은 effective batch가 커질수록 일반적으로 비례 증가가 필요합니다 (linear scaling rule).
- PyTorch 학습 루프에서는 위 값을 직접 곱해 lr을 조정합니다. Lightning `Trainer`나 HF `TrainingArguments`는 인자만 받습니다.

## rank 간 metric 합산 (AverageMeter)

DDP에서 검증 loss/accuracy를 측정할 때, 각 rank는 자기 몫의 데이터만 봅니다. 전체 평균을 구하려면 `dist.all_reduce(SUM)`로 rank별 합과 카운트를 모은 뒤 driver(rank 0)에서 나눠야 합니다.

```python
import torch
import torch.distributed as dist


class AverageMeter:
    """rank별 누적치를 들고 있다가 all_reduce로 전체 평균을 만듭니다."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0

    def update(self, loss_sum: float, correct: int, total: int):
        self.loss_sum = loss_sum
        self.correct = correct
        self.total = total

    def all_reduce(self, device) -> tuple[float, float]:
        """모든 rank의 (loss_sum, correct, total)을 합산하고 평균/정확도를 반환."""
        t = torch.tensor([self.loss_sum, self.correct, self.total], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=False)
        loss_sum, correct, total = t.tolist()
        return loss_sum / total, correct / total

    def reduce(self) -> tuple[float, float]:
        """non-distributed 컨텍스트용 fallback."""
        return self.loss_sum / self.total, self.correct / self.total
```

사용 예 (eval 루프):

```python
meter = AverageMeter()
loss_sum, correct, total = 0.0, 0, 0
for batch in val_loader:
    # ... forward, loss, prediction ...
    loss_sum += loss.item() * batch_size
    correct += (preds == labels).sum().item()
    total += labels.size(0)
    meter.update(loss_sum, correct, total)

avg_loss, avg_acc = meter.all_reduce(device)   # rank 간 합산 후 평균
dist.barrier()                                  # 모든 rank가 다음 단계 전에 대기
if rank == 0:
    mlflow.log_metric("val/loss", avg_loss)
    mlflow.log_metric("val/acc", avg_acc)
```

> 주의: `all_reduce`는 모든 rank가 같은 시점에 호출해야 합니다. rank 0에서만 호출하면 다른 rank가 hang합니다.

## 참고

- PyTorch DDP 가이드: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- Databricks 분산 학습 개요: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/
