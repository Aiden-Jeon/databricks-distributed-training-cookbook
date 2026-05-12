# Recommender Baseline (Two-Tower MLP)

본 쿡북이 공유하는 **기준 모델**. 분산 학습 패턴 자체에 집중하기 위해 모델 정의는 한 곳에 모아두고, 노트북 셀(1×1 / 1×N / M×N)은 같은 모델·같은 데이터셋을 **launcher 설정만 바꿔** 실행합니다.

## 데이터셋

[MovieLens 25M](https://files.grouplens.org/datasets/movielens/ml-25m.zip) (GroupLens, ~250MB 압축). user-movie rating 25M 행. implicit feedback으로 변환:

- positive: `rating >= 4` (≈ 12M 행)
- negative: 각 positive마다 해당 user가 아직 보지 않은 movie를 uniform 랜덤 1개 샘플링 (negative ratio 1:1)
- label ∈ {0, 1}, BCE loss

userId/movieId는 dense index(0-based contiguous)로 remap합니다. embedding lookup 효율 + 모델 vocab 크기 명시화.

| 항목 | 값 |
|------|-----|
| 원본 행 수 | 25,000,095 |
| positive (rating>=4) | ≈ 12,580,000 |
| 학습 행 수 (pos+neg) | ≈ 25,160,000 |
| n_users | 162,541 |
| n_items | 59,047 (실제 등장 movie 수) |

다운로드와 전처리는 [`data-pipeline.md`](data-pipeline.md), 그리고 각 행의 `01-data_prep.ipynb`에서 처리합니다.

## 모델 구조

User ID와 Item ID를 입력받아 클릭(=긍정 interaction) 확률을 예측하는 dual-encoder.

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

손실: `BCEWithLogitsLoss`. 학습 데이터는 `(user_id, item_id, label∈{0,1})` 튜플.

## 단일 config

scale 매트릭스(small/medium/large) 대신 ML-25M 단일 데이터셋을 기준으로 **고정된 하이퍼파라미터**를 사용합니다. 1×1 / 1×N / M×N 토폴로지 차이는 모델·데이터가 아니라 **launcher 설정과 batch_size에 의해 결정**됩니다.

| 항목 | 값 |
|------|----|
| n_users | 162,541 |
| n_items | 59,047 |
| emb_dim | 64 |
| tower_hidden | (256, 128) |
| 학습 파라미터 (대략) | ~15M (대부분 임베딩 테이블) |
| 학습 행 수 | ≈ 25M (pos+neg) |
| 1×1 / 1×N / M×N batch_size | 4096 / 8192 / 16384 (global, 노드·GPU 수에 맞춰 키움) |

> 모델은 A10G 24GB 단일 GPU에 충분히 들어갑니다. M×N의 정당화는 **모델 크기가 아니라 데이터 처리량 확장**입니다. 더 큰 임베딩이 필요해지면 GPU 메모리가 더 큰 인스턴스(A100, H100)나 임베딩 샤딩(TorchRec, 본 쿡북 범위 밖)이 필요합니다.

## 손실·옵티마이저 (모든 토폴로지 공통)

```python
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

학습 epoch 수는 `num_epochs=10` 상한, 실제 종료는 `EarlyStopping(patience=3, min_delta=1e-4)`이 결정. 15분 budget 안에 끝내기 위해 `max_steps_per_epoch=200`으로 epoch당 step을 캡 합니다.

## 평가 지표

| 지표 | 의미 |
|------|------|
| AUC | 무작위 양/음 샘플 분류 능력 |
| Hit@K | top-K 안에 실제 클릭 아이템이 포함될 확률 |
| NDCG@K | 순위 가중을 둔 hit 점수 |

본 쿡북에서는 학습 중 **val/loss(BCE)** 를 metric으로 logging합니다. 실제 추천 도메인 metric(AUC, Hit@K, NDCG@K)은 inference 단계 외부 작업으로 둡니다. mock 데이터와 달리 ML-25M에서는 val/loss가 학습 진행에 따라 의미 있게 감소합니다.

## 참고

- 모델 자체는 단순하지만, 본 쿡북의 목적은 **분산 학습 launcher×topology 매트릭스를 익히는 것**입니다.
- 추천 도메인의 더 큰 모델(SASRec, BERT4Rec, GNN 기반)이 필요해지면 같은 패턴에 모델만 교체합니다.
- MovieLens 라이선스: 비상업적 연구 목적. https://files.grouplens.org/datasets/movielens/ml-25m-README.html
