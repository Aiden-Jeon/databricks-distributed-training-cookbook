# Recommender Baseline (Two-Tower MLP)

본 쿡북의 12개 셀이 공유하는 **기준 모델**. 분산 학습 패턴 자체에 집중하기 위해 모델 정의는 한 곳에 모아두고, 셀별로 **하이퍼파라미터(임베딩 크기, 사용자/아이템 수)만 스케일**한다.

## 모델 구조

User ID와 Item ID를 입력받아 클릭 확률(또는 평점)을 예측하는 가장 단순한 dual-encoder 추천 모델.

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

손실: implicit feedback 가정 → `BCEWithLogitsLoss`. 학습 데이터는 `(user_id, item_id, label∈{0,1})` 튜플의 묶음.

## 셀별 스케일

같은 모델 골격에 **id 수와 embedding dim만** 키운다. 분산 학습이 정당화될 만큼 파라미터 수를 늘리는 게 목적이다.

| 셀 | user 수 | item 수 | emb dim | tower hidden | 학습 파라미터 (대략) | 15분 budget 가정 |
|----|---------|---------|---------|--------------|--------------------|-----------------|
| 1×1 | 100K | 50K | 64 | (256, 128) | ~10M | 5~10분 |
| 1×N | 1M | 500K | 128 | (512, 256) | ~200M | 10분 |
| M×N | 10M | 5M | 256 | (1024, 512) | ~수 GB | 15분 |

> 파라미터의 대부분은 임베딩 테이블에서 온다. M×N 셀에서는 임베딩 테이블이 수 GB가 되어, 단일 GPU에 들어가더라도 **데이터 처리량을 노드 수로 확장**하는 multi-node DDP가 의미를 가진다.

## 손실·옵티마이저 (모든 셀 공통)

```python
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

학습 epoch 수는 `epochs=1`을 기본으로 한다 (15분 budget 안에 끝내기 위함). 더 큰 셀에서는 step 수를 명시적으로 제한한다.

## 평가 지표

추천 도메인 표준 지표 중 가벼운 것 위주.

| 지표 | 의미 |
|------|------|
| AUC | 무작위 양/음 샘플 분류 능력 |
| Hit@K | top-K 안에 실제 클릭 아이템이 포함될 확률 |
| NDCG@K | 순위 가중을 둔 hit 점수 |

본 쿡북 셀의 `eval_and_register` 단계에서는 AUC만 본다. K-기반 지표는 inference 단계 외부 작업으로 둔다.

## 참고

- 모델 자체는 단순하지만, 본 쿡북의 목적은 **분산 학습 launcher×topology 매트릭스를 익히는 것**이다.
- 추천 도메인의 더 큰 모델(SASRec, BERT4Rec, GNN 기반)이 필요해지면 같은 패턴에 모델만 교체한다.
