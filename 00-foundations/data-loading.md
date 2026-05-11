# Data Loading

분산 학습에서의 데이터 로딩 패턴. 본 쿡북은 추천(user-item interaction) 시나리오를 가정합니다.

## 본 쿡북의 기본 선택: MovieLens 25M

쿡북 셀들의 학습 시간은 **15분 이내**로 제한합니다. *분산 학습 패턴 자체*에 집중하면서도 **val/loss가 의미 있게 감소하는** 학습을 보여주기 위해 [MovieLens 25M](https://files.grouplens.org/datasets/movielens/ml-25m.zip) (GroupLens, ~250MB) 실데이터를 사용합니다. 합성 데이터로는 랜덤 라벨 때문에 metric이 움직이지 않습니다.

자세한 데이터 스키마와 implicit feedback 변환 규칙은 [`recommender-baseline.md`](recommender-baseline.md). 핵심 요약:

| 단계 | 처리 |
|------|------|
| 다운로드 | `ml-25m.zip` → UC Volume에 unzip |
| 라벨 변환 | `rating >= 4` → positive (label=1) |
| negative sampling | positive 당 1개, uniform 랜덤 movie (이미 본 것 제외 없이 간단히) |
| index remap | userId, movieId → 0-based contiguous (embedding lookup용) |
| split | 행 단위 random 10% → val |
| 저장 | `train/`, `val/` 각각 `shard-XXXXX.parquet` 묶음으로 UC Volume에 |

## 다운로드 + 압축 해제

```python
import os
import urllib.request
import zipfile

ML25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
local_zip = "/local_disk0/ml-25m.zip"
extract_root = "/local_disk0/ml-25m-extract"

urllib.request.urlretrieve(ML25M_URL, local_zip)
with zipfile.ZipFile(local_zip) as zf:
    zf.extractall(extract_root)
ratings_csv = os.path.join(extract_root, "ml-25m", "ratings.csv")
```

`/local_disk0`는 노드 로컬 디스크입니다. UC Volume에 바로 zip을 풀어도 되지만 zip 해제 I/O가 느려서 driver 로컬 디스크 → UC Volume 순으로 옮기는 것이 빠릅니다.

## 라벨 변환 + negative sampling + remap

```python
import numpy as np
import pandas as pd

ratings = pd.read_csv(ratings_csv)  # ~600MB in pandas
pos = ratings.loc[ratings["rating"] >= 4.0, ["userId", "movieId"]].copy()
pos["label"] = np.float32(1.0)

# dense remap: userId, movieId → 0..N-1 contiguous index
unique_users = np.sort(ratings["userId"].unique())
unique_items = np.sort(ratings["movieId"].unique())
user_to_idx = {u: i for i, u in enumerate(unique_users)}
item_to_idx = {m: i for i, m in enumerate(unique_items)}
pos["user_id"] = pos["userId"].map(user_to_idx).astype(np.int64)
pos["item_id"] = pos["movieId"].map(item_to_idx).astype(np.int64)

# negative sampling: positive 당 1개, uniform 랜덤 movie
rng = np.random.default_rng(0)
n_pos = len(pos)
neg = pd.DataFrame({
    "user_id": pos["user_id"].to_numpy(),
    "item_id": rng.integers(0, len(unique_items), size=n_pos, dtype=np.int64),
    "label": np.zeros(n_pos, dtype=np.float32),
})

interactions = pd.concat([pos[["user_id", "item_id", "label"]], neg], ignore_index=True)
interactions = interactions.sample(frac=1.0, random_state=0).reset_index(drop=True)
n_users = len(unique_users)
n_items = len(unique_items)
```

> negative sampling 정확도: positive를 "이미 본 movie"에서 제외하지 않는 단순 uniform 샘플링. ML-25M density는 25M / (162K × 59K) ≈ 0.26%로 매우 sparse하기 때문에 negative로 뽑은 movie가 우연히 그 user의 positive일 확률은 무시 가능합니다. 정밀한 추천 평가가 필요하면 user별 positive set을 제외하는 rejection sampling이 필요하지만, 본 쿡북은 분산 학습 메커닉 데모가 목적이므로 단순화합니다.

## shard 단위 parquet write (대용량 안전)

전체 ~25M 행을 driver에 한 번에 올리는 것은 무겁습니다. 행 단위 shard로 잘라 `train/`과 `val/`에 나누어 쓰면 학습 노트북이 `pq.read_table(files)`로 묶어 읽을 수 있습니다.

```python
import os
import pyarrow as pa
import pyarrow.parquet as pq

def write_sharded(df, out_dir, val_ratio=0.1, rows_per_shard=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "val"), exist_ok=True)
    for shard_idx, start in enumerate(range(0, len(df), rows_per_shard)):
        chunk = df.iloc[start : start + rows_per_shard]
        is_val = rng.random(size=len(chunk)) < val_ratio
        for split, mask in [("train", ~is_val), ("val", is_val)]:
            sub = chunk.loc[mask, ["user_id", "item_id", "label"]]
            if len(sub) == 0:
                continue
            table = pa.table({
                "user_id": sub["user_id"].to_numpy(),
                "item_id": sub["item_id"].to_numpy(),
                "label": sub["label"].to_numpy(),
            })
            pq.write_table(table, os.path.join(out_dir, split, f"shard-{shard_idx:05d}.parquet"))
```

생성 결과 디렉토리 구조:
```
<DATA_DIR>/
  train/shard-00000.parquet
  train/shard-00001.parquet
  ...
  val/shard-00000.parquet
  ...
```

## 학습 노트북의 데이터 로딩

학습 코드는 `train/`과 `val/`을 각각 읽어 `TensorDataset`을 만듭니다.

```python
import os
import pyarrow.parquet as pq
import torch
from torch.utils.data import TensorDataset


def load_split(split_dir):
    files = sorted(
        os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".parquet")
    )
    table = pq.read_table(files, columns=["user_id", "item_id", "label"])
    return TensorDataset(
        torch.from_numpy(table.column("user_id").to_numpy()),
        torch.from_numpy(table.column("item_id").to_numpy()),
        torch.from_numpy(table.column("label").to_numpy()),
    )

train_dataset = load_split(os.path.join(DATA_DIR, "train"))
val_dataset = load_split(os.path.join(DATA_DIR, "val"))
```

DDP에서는 `torch.utils.data.distributed.DistributedSampler`로 rank별로 자동 분할합니다.

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, drop_last=True)
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, pin_memory=True)
```

## 메모리 가이드

`pq.read_table`은 전체 split을 RAM에 올립니다. ML-25M의 train split(~22M 행, 컬럼 3개 int/float)은 약 600MB이므로 driver와 각 worker GPU 노드에서 충분히 들어갑니다. 더 큰 데이터에서는 `pq.ParquetDataset` + lazy `__getitem__` 또는 `petastorm` 패턴을 고려합니다.

## 참고

- MovieLens 25M README: https://files.grouplens.org/datasets/movielens/ml-25m-README.html
- 분산 학습 데이터 준비: https://docs.databricks.com/aws/en/machine-learning/load-data/ddl-data
- `DistributedSampler`: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
