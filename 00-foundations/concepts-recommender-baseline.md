# Recommender Baseline (Two-Tower MLP)

본 쿡북이 공유하는 **기준 모델**을 정의합니다. 분산 학습 패턴 자체에 집중할 수 있도록 모델 정의는 이 문서 한 곳에 모아 두고, 노트북 셀(1×1 / 1×N / M×N)은 같은 모델과 같은 데이터셋을 **launcher 설정만 바꿔** 실행합니다.

## 데이터셋

데이터셋은 [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)(GroupLens, 약 250MB 압축)을 사용합니다. user-movie rating 25M 행을 implicit feedback으로 변환합니다.

- positive: `rating >= 4` (약 12M 행)
- negative: 각 positive마다 해당 user가 아직 보지 않은 movie를 uniform 랜덤 1개 샘플링 (negative ratio 1:1)
- label ∈ {0, 1}, BCE loss

userId/movieId는 dense index(0-based contiguous)로 remap합니다. embedding lookup 효율을 높이고 모델 vocab 크기를 명시적으로 드러내기 위함입니다.

| 항목 | 값 |
|------|-----|
| 원본 행 수 | 25,000,095 |
| positive (rating>=4) | ≈ 12,580,000 |
| 학습 행 수 (pos+neg) | ≈ 25,160,000 |
| n_users | 162,541 |
| n_items | 59,047 (실제 등장 movie 수) |

표는 ML-25M을 implicit feedback으로 변환했을 때 학습에 실제로 들어가는 행 수와 vocab 크기를 정리한 것입니다. 다운로드와 전처리는 [`data-pipeline.md`](data-pipeline.md), 그리고 각 행의 `01-data_prep.ipynb`에서 처리합니다.

## 모델 구조

User ID와 Item ID를 입력받아 클릭(긍정 interaction) 확률을 예측하는 dual-encoder입니다.

```python
import torch
import torch.nn as nn


class TwoTowerMLP(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        tower_hidden: tuple[int, ...] = (256, 128),
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.user_tower = self._make_tower(emb_dim, tower_hidden)
        self.item_tower = self._make_tower(emb_dim, tower_hidden)

    @staticmethod
    def _make_tower(in_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        return nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_tower(self.user_emb(user_ids))
        i = self.item_tower(self.item_emb(item_ids))
        return (u * i).sum(dim=-1)  # logit
```

손실 함수는 `BCEWithLogitsLoss`를 쓰고, 학습 데이터는 `(user_id, item_id, label∈{0,1})` 튜플입니다.

## 쿡북에서 다루는 분산 학습 전략: DDP

분산 학습 전략은 여러 가지가 있지만, 본 쿡북은 **DDP(DistributedDataParallel) 한 가지만** 다룹니다. 각 GPU에 모델을 통째로 복제해 두고 데이터를 나눠 처리하는 가장 기본적인 데이터 병렬 방식입니다.

| 항목 | DDP (DistributedDataParallel) |
|------|-------------------------------|
| 모델 메모리 | 각 GPU에 모델 **전체** 복제 |
| 그래디언트 | 각 GPU 계산 후 AllReduce |
| 데이터 | rank별로 미니배치를 나눠 처리 |
| 통신량 | 작음 (AllReduce 1회/스텝) |
| 적합 모델 크기 | 한 GPU에 모델 + optimizer state가 들어가는 크기 |

위 모델은 단일 GPU에 충분히 올라가므로 DDP만으로 충분합니다.

## 단일 config

scale 매트릭스(small/medium/large) 대신 ML-25M 단일 데이터셋을 기준으로 **고정된 하이퍼파라미터**를 사용합니다. 1×1 / 1×N / M×N 토폴로지 차이는 모델이나 데이터가 아니라 **launcher 설정과 batch_size로만** 결정됩니다.

| 항목 | 값 |
|------|----|
| n_users | 162,541 |
| n_items | 59,047 |
| emb_dim | 64 |
| tower_hidden | (256, 128) |
| 학습 파라미터 (대략) | ~15M (대부분 임베딩 테이블) |
| 학습 행 수 | ≈ 25M (pos+neg) |
| 1×1 / 1×N / M×N batch_size | 4096 / 8192 / 16384 (global, 노드·GPU 수에 맞춰 키움) |

표에서 보듯 모델은 A10G 24GB 단일 GPU에 충분히 들어갑니다. M×N을 다루는 이유는 모델 크기가 아니라 **데이터 처리량 확장** 때문입니다. 더 큰 임베딩이 필요해지면 GPU 메모리가 더 큰 인스턴스(A100, H100)나 임베딩 샤딩(TorchRec, 본 쿡북 범위 밖)이 필요합니다.

## 손실·옵티마이저 (모든 토폴로지 공통)

```python
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

학습 epoch 수는 `num_epochs=10`을 상한으로 두고, 실제 종료 시점은 `EarlyStopping(patience=3, min_delta=1e-4)`이 결정합니다. 15분 budget 안에 끝내기 위해 `max_steps_per_epoch=200`으로 epoch당 step을 캡합니다.

## 평가 지표

추천 도메인에서 흔히 보는 지표를 정리하면 다음과 같습니다.

| 지표 | 의미 |
|------|------|
| AUC | 무작위 양/음 샘플 분류 능력 |
| Hit@K | top-K 안에 실제 클릭 아이템이 포함될 확률 |
| NDCG@K | 순위 가중을 둔 hit 점수 |

본 쿡북에서는 학습 중 **val/loss(BCE)** 를 metric으로 logging합니다. 실제 추천 도메인 metric(AUC, Hit@K, NDCG@K)은 inference 단계의 외부 작업으로 두고 다루지 않습니다. mock 데이터와 달리 ML-25M에서는 val/loss가 학습 진행에 따라 의미 있게 감소합니다.

## rank 간 metric 합산 (AverageMeter)

DDP 환경에서 검증 loss나 accuracy를 측정할 때 흔히 빠뜨리는 부분입니다. 각 rank는 자기 몫의 데이터만 보기 때문에, rank별 평균을 그대로 출력하면 전체 검증 셋 평균과 어긋납니다. 올바른 전체 평균을 구하려면 `dist.all_reduce(SUM)`으로 rank별 합과 카운트를 모은 뒤 driver(rank 0)에서 나눠야 합니다. 본 쿡북은 다음과 같은 `AverageMeter` 패턴을 표준으로 사용합니다.

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

실제 eval 루프에 끼워 쓰는 모양은 다음과 같습니다.

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

> 주의: `all_reduce`는 모든 rank가 같은 시점에 호출해야 합니다. rank 0에서만 호출하면 나머지 rank들이 그 위치에서 영영 기다리며 멈춰 버립니다.

## 참고

모델 자체는 단순하지만 본 쿡북의 목적은 **분산 학습 launcher × topology 매트릭스를 익히는 것**입니다. 추천 도메인의 더 큰 모델(SASRec, BERT4Rec, GNN 기반)이 필요해지면 같은 패턴 위에서 모델만 교체하면 됩니다.

자세한 내용은 다음 자료를 참조하세요.

- [MovieLens 25M README (라이선스: 비상업적 연구 목적)](https://files.grouplens.org/datasets/movielens/ml-25m-README.html)
- [PyTorch DDP 가이드](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Databricks 분산 학습 개요](https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/)
