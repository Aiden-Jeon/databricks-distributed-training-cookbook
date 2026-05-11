"""Two-Tower MLP 모델 정의와 학습 보조 클래스.

02-script-based의 모든 trainer (TorchDistributor / Lightning / Accelerate)가
공통으로 import한다. 기준 모델 정의는 00-foundations/recommender-baseline.md.
"""

import torch
import torch.nn as nn


class TwoTowerMLP(nn.Module):
    """User tower와 Item tower의 inner product로 score를 내는 단순 Two-Tower 추천 모델.

    토폴로지(1x1 / 1xN / MxN)와 무관하게 동일한 모델 구조를 사용한다.
    스케일(batch size, world size 등)은 호출 측에서 결정.
    """

    def __init__(self, n_users, n_items, emb_dim, tower_hidden):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.user_tower = self._make_tower(emb_dim, tower_hidden)
        self.item_tower = self._make_tower(emb_dim, tower_hidden)

    @staticmethod
    def _make_tower(in_dim, hidden):
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        return nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        u = self.user_tower(self.user_emb(user_ids))
        i = self.item_tower(self.item_emb(item_ids))
        return (u * i).sum(dim=-1)


class EarlyStopping:
    """val_loss 기준 patience 카운터. 모든 rank가 동일한 결정에 도달하도록
    호출자는 dist.all_reduce로 합산된 동일한 val_loss를 모든 rank에서 step()에 넘긴다.
    """

    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
