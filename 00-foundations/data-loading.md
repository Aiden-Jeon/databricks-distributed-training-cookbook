# Data Loading

분산 학습에서의 데이터 로딩 패턴. 본 쿡북은 추천(user-item interaction) 시나리오를 가정한다.

## 본 쿡북의 기본 선택: 합성 데이터

쿡북 셀들의 학습 시간은 **15분 이내**로 제한한다. 실제 데이터 전처리·다운로드 시간을 빼고 *분산 학습 패턴 자체*에 집중하기 위해 합성 user-item interaction 데이터를 사용한다.

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset


def make_synthetic_interactions(
    n_users: int,
    n_items: int,
    n_rows: int,
    pos_ratio: float = 0.1,
    seed: int = 0,
) -> TensorDataset:
    rng = np.random.default_rng(seed)
    user_ids = rng.integers(0, n_users, size=n_rows, dtype=np.int64)
    item_ids = rng.integers(0, n_items, size=n_rows, dtype=np.int64)
    labels = (rng.random(size=n_rows) < pos_ratio).astype(np.float32)
    return TensorDataset(
        torch.from_numpy(user_ids),
        torch.from_numpy(item_ids),
        torch.from_numpy(labels),
    )
```

셀별 행 수는 [`cluster-recipes.md`](cluster-recipes.md)의 토폴로지 표를 따른다. 합성 데이터 생성은 driver에서 한 번 만들어 `/Volumes/...`에 parquet로 저장하면 재실행 시 GPU 시간을 아낄 수 있다.

## 패턴 1: 합성 → Delta → HF Dataset (1×1, 1×N 셀)

```python
import datasets
import pandas as pd

pdf = pd.DataFrame({"user_id": ..., "item_id": ..., "label": ...})
spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable(
    "main.recsys.interactions_small"
)

ds = datasets.Dataset.from_pandas(
    spark.table("main.recsys.interactions_small").toPandas()
)
```

- 장점: 코드가 짧고 MLflow에 데이터 lineage가 자동 기록된다.
- 한계: 전체를 driver 메모리에 올린다. 1×1·1×N 셀까지는 충분.

## 패턴 2: parquet → torch DataLoader (M×N 셀)

거대 합성 데이터(예: 1억 행)는 parquet shard로 UC Volume에 두고, 각 worker가 자기 몫만 읽는다.

```python
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset


class ParquetInteractionDataset(Dataset):
    def __init__(self, path: str):
        self.table = pq.read_table(path)

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, idx: int):
        row = self.table.slice(idx, 1).to_pylist()[0]
        return row["user_id"], row["item_id"], row["label"]


loader = DataLoader(
    ParquetInteractionDataset("/Volumes/main/recsys/interactions_large"),
    batch_size=4096,
    sampler=DistributedSampler(...),   # DDP에서 rank별 분할
    num_workers=4,
)
```

DDP에서는 `torch.utils.data.distributed.DistributedSampler`로 rank별로 데이터를 자동 분할한다.

## Spark → torch loader (실데이터 연결 시)

실제 Delta 테이블이 driver 메모리에 들어가지 않는 규모면 `petastorm` 또는 `databricks.ml.streaming` 패턴으로 spark dataframe을 직접 torch DataLoader에 연결한다. 본 쿡북은 합성 데이터로 시작하므로 이 패턴은 셀별 README에서만 참고로 언급한다.

## 참고

- 분산 학습 데이터 준비: https://docs.databricks.com/aws/en/machine-learning/load-data/ddl-data
- `DistributedSampler`: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
