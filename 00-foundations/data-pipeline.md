# Data Loading

이 문서는 분산 학습에서의 데이터 로딩 패턴을 정리합니다. 본 쿡북은 추천(user-item interaction) 시나리오를 가정합니다.

## 본 쿡북의 기본 선택: MovieLens 25M

쿡북 셀들의 학습 시간은 **15분 이내**로 제한합니다. *분산 학습 패턴 자체*에 집중하면서도 **val/loss가 의미 있게 감소하는** 학습을 보여주기 위해 [MovieLens 25M](https://files.grouplens.org/datasets/movielens/ml-25m.zip) (GroupLens, ~250MB) 실데이터를 사용합니다. 합성 데이터를 쓰면 라벨이 랜덤이라 metric이 움직이지 않기 때문입니다.

자세한 데이터 스키마와 implicit feedback 변환 규칙은 [`concepts-recommender-baseline.md`](concepts-recommender-baseline.md)에 있습니다. 핵심을 한 표로 요약하면 다음과 같습니다.

| 단계 | 처리 |
|------|------|
| 다운로드 | `ml-25m.zip` → UC Volume에 unzip |
| 라벨 변환 | `rating >= 4` → positive (label=1) |
| negative sampling | positive 당 1개, uniform 랜덤 movie (이미 본 것 제외 없이 간단히) |
| index remap | userId, movieId → 0-based contiguous (embedding lookup용) |
| split | 행 단위 random 10% → val |
| 저장 | `train/`, `val/` 각각 `shard-XXXXX.parquet` 묶음으로 UC Volume에 |

## 다운로드 + 압축 해제

먼저 driver의 로컬 디스크에 zip을 받아 풉니다.

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

`/local_disk0`은 노드 로컬 디스크입니다. UC Volume에 바로 zip을 풀어도 되지만 zip 해제 I/O가 느리므로, driver 로컬 디스크에서 푼 뒤 UC Volume으로 옮기는 편이 빠릅니다.

## 라벨 변환 + negative sampling + remap

implicit feedback 변환과 인덱스 remap, negative sampling을 한 번에 처리합니다.

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

> negative sampling 정확도에 관한 메모입니다. 위 코드는 positive를 "이미 본 movie"에서 제외하지 않는 단순 uniform 샘플링입니다. ML-25M density는 25M / (162K × 59K) ≈ 0.26%로 매우 sparse하기 때문에 negative로 뽑은 movie가 우연히 그 user의 positive일 확률은 무시할 수 있습니다. 정밀한 추천 평가가 필요하면 user별 positive set을 제외하는 rejection sampling을 써야 하지만, 본 쿡북은 분산 학습 메커닉 데모가 목적이므로 단순화합니다.

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

생성 결과 디렉토리 구조는 다음과 같습니다.

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

`pq.read_table`은 전체 split을 RAM에 올립니다. ML-25M의 train split(~22M 행, 컬럼 3개 int/float)은 약 600MB이므로 driver와 각 worker GPU 노드에 충분히 들어갑니다. 더 큰 데이터에서는 `pq.ParquetDataset` + lazy `__getitem__` 또는 `petastorm` 패턴을 고려합니다.

## Databricks 경로 비교 — 어디에 데이터를 두는가

분산 학습에서 데이터 경로 선택은 throughput과 안정성에 직결됩니다. 후보별 특성을 한눈에 비교하면 다음과 같습니다.

| 경로 | driver 접근 | worker 접근 | 영속성 | read 성능 | 권장 용도 |
|------|------------|------------|--------|-----------|----------|
| `/local_disk0/...` | ✅ | ✅ (각 노드의 로컬 SSD) | 클러스터 종료 시 소실 | 가장 빠름 (NVMe) | 임시 unzip, scratch space |
| DBFS (`/dbfs/...`) | ✅ | ✅ | 영속 | 중간 (FUSE 오버헤드) | 레거시 — 신규 사용 비권장 |
| **UC Volume** (`/Volumes/.../`) | ✅ | ✅ | 영속, UC governance | 중간 (cloud object store) | 본 쿡북 기본 |
| Cloud direct (`s3://`, `abfss://`) | API 호출 | API 호출 | 영속 | 가장 빠른 cloud read (UC Volume 우회) | TB+ 데이터, throughput 최우선 |

### `/local_disk0`의 노드별 분리

`/local_disk0`은 **각 노드의 로컬 SSD**입니다. driver의 `/local_disk0`과 worker의 `/local_disk0`은 서로 다른 디스크라는 점을 기억해야 합니다. 따라서 다음과 같은 특성이 따라옵니다.

- driver에서 unzip하고 같은 `/local_disk0`에 데이터를 두면 driver만 학습하는 1×1, 1×N에서는 빠르게 동작합니다.
- multi-node(M×N) 학습은 worker가 driver의 `/local_disk0`을 볼 수 없기 때문에 UC Volume으로 옮겨야 합니다.
- 클러스터 restart나 auto-termination 시 모든 노드의 `/local_disk0`이 함께 소실됩니다.

본 쿡북의 `01-data_prep.ipynb`가 `/local_disk0`에 unzip한 뒤 UC Volume에 shard parquet로 옮기는 이유가 여기에 있습니다. driver-only 작업은 빠른 로컬을 쓰고, 학습 데이터는 worker에서도 보이는 UC Volume에 두는 분업입니다.

### UC Volume read 성능

UC Volume은 내부적으로 cloud object store(S3/ADLS/GCS)를 마운트한 것이므로 다음과 같은 성능 특성을 보입니다.

- 첫 read는 cloud latency가 그대로 더해져 수십 ms 수준입니다.
- 같은 노드에서 같은 파일을 다시 읽으면 Databricks Runtime의 **disk cache**가 SSD에 저장되어 두 번째 read부터 빨라집니다.
- 노드 간에는 cache가 공유되지 않으므로 multi-node에서 모든 rank가 같은 train split을 읽으면 각 worker가 독립적으로 캐싱합니다.

ML-25M의 600MB train split은 첫 epoch의 첫 read에 몇 초가 걸리고, 이후로는 본 쿡북이 `pq.read_table`로 일괄 RAM에 로드하기 때문에 disk cache 여부와 무관해집니다.

### multi-rank 동시 읽기 시 contention

같은 노드의 여러 rank(예: 1×N에서 4개 rank)가 같은 parquet 파일을 동시에 `pq.read_table`하면 어떻게 되는지 정리합니다.

- 첫 rank가 캐시를 채우고, 나머지는 캐시 hit로 읽으므로 contention이 거의 없습니다.
- 다른 노드의 rank들과는 독립적으로 동작합니다.

DDP에서 `DistributedSampler`가 인덱스만 나누고 각 rank가 같은 파일을 전부 로드하는 본 쿡북 패턴은 ML-25M 규모에서는 문제가 되지 않습니다. **TB+ 데이터에서는 rank별로 다른 shard만 읽도록** 분배해야 하며 petastorm, mosaic-streaming 같은 라이브러리가 그 역할을 합니다.

### cloud direct read

UC Volume을 거치지 않고 `s3://my-bucket/...`로 직접 읽는 옵션의 트레이드오프는 다음과 같습니다.

- 장점은 FUSE 오버헤드를 우회해 최대 throughput을 낼 수 있다는 점입니다.
- 단점으로는 UC governance·lineage가 끊기고 IAM 자격증명을 별도로 설정해야 합니다.
- Databricks disk cache가 적용되지 않습니다.

`pq.read_table`은 s3 경로를 그대로 받아 pyarrow가 boto3/s3fs로 호출합니다. 본 쿡북 규모에서는 UC Volume으로 충분합니다.

### 의사결정 트리

데이터 크기와 토폴로지에 따라 어떤 경로를 고를지 정리하면 다음과 같습니다.

```
< 1GB, 모든 토폴로지
  → UC Volume (본 쿡북 기본)

1GB ~ 100GB, multi-node
  → UC Volume + disk cache 신뢰 (각 노드에 자동 캐싱)

100GB+, throughput critical
  → cloud direct (s3://) + shard별 rank 분배 (petastorm/mosaic-streaming)

학습 중 임시 파일 (unzip, intermediate)
  → /local_disk0 (driver only) 또는 클러스터 종료까지만 필요한 데이터
```

## 참고

자세한 내용은 다음 자료를 참조하세요.

- [MovieLens 25M README](https://files.grouplens.org/datasets/movielens/ml-25m-README.html)
- [Databricks: 분산 학습 데이터 준비](https://docs.databricks.com/aws/en/machine-learning/load-data/ddl-data)
- [PyTorch `DistributedSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
