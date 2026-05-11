# 01-1 · 노트북 only · 1×1 GPU

> 노트북 셀 안에서 PyTorch 학습 루프를 직접 호출하는 가장 단순한 시작점. Two-Tower MLP 추천 모델을 단일 GPU에서 한 epoch 학습한다.

## 🎯 시나리오

- Two-Tower MLP의 가장 작은 변형 (~10M params, [recommender-baseline](../../00-foundations/recommender-baseline.md))
- PoC, 데모, 또는 단일 GPU로도 충분한 작은 데이터셋
- 코드를 파일로 분리하지 않고 노트북에서 빠르게 반복하고 싶은 단계

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (user 100K × item 50K, emb dim 64) |
| 라이브러리 | `torch`, `mlflow` |
| 병렬화 | 단일 GPU (분산화 없음) |
| 데이터 | 합성 user-item interaction → Delta → torch DataLoader |
| 실행 | 노트북 셀에서 학습 루프 직접 호출 |
| 추적 | MLflow autolog (`mlflow.pytorch.autolog()`) |
| 목표 학습 시간 | 5~10분 |

## 🖥️ 클러스터 권장 사양

| 인스턴스 | GPU |
|---------|-----|
| `g5.2xlarge` | 1× A10G 24GB |

- 클라우드: AWS only (Azure/GCP는 본 쿡북 스코프 밖)
- DBR: 15.x ML GPU 또는 Serverless GPU 워크스페이스
- AI Runtime: 불필요 (`@distributed` 미사용)

## 📂 파일

```
01-single-node-single-gpu/
├── README.md
├── 00_setup.py              # 라이브러리 설치, 경로/하이퍼파라미터 변수
├── 01_data_prep.py          # 합성 interaction 데이터 → Delta
├── 02_train.py              # Two-Tower MLP 학습 루프
└── 03_eval_and_register.py  # 평가 + UC Model Registry 등록
```

## 🚀 실행 순서

1. `00_setup.py` 실행 → 라이브러리 설치, `CATALOG`/`SCHEMA`/`N_USERS`/`N_ITEMS` 변수 설정.
2. `01_data_prep.py` 실행 → 합성 interaction을 Delta 테이블로 저장.
3. `02_train.py` 실행 → MLflow run 시작 → 학습 루프 → 모델 state_dict 저장.
4. `03_eval_and_register.py` 실행 → AUC 측정 → UC에 등록.

## 🧬 핵심 패턴

```python
import mlflow
import torch
from torch.utils.data import DataLoader

mlflow.pytorch.autolog()
device = "cuda"

with mlflow.start_run(run_name="recommender-1x1"):
    model = TwoTowerMLP(N_USERS, N_ITEMS, emb_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for users, items, labels in DataLoader(train_ds, batch_size=4096, shuffle=True):
        users, items, labels = users.to(device), items.to(device), labels.to(device)
        logits = model(users, items)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
```

## ⚠️ 함정

- BF16 mixed precision은 Ampere(A10G/A100) 이상에서만. T4면 FP16 + `GradScaler`.
- 임베딩 테이블은 GPU 메모리의 큰 비중을 차지한다 → 배치 크기는 작게 시작해 점진적으로 증가.
- 체크포인트는 `/local_disk0/`에 쓰고 학습 종료 후 `/Volumes/...`로 복사 ([uc-volumes](../../00-foundations/uc-volumes-checkpoints.md)).

## ➡️ 다음 셀

- 옆 (같은 실행 방식, 더 큰 토폴로지): [01-2 · 노트북 · 1×N GPU](../02-single-node-multi-gpu/)
- 아래 (같은 토폴로지, 다른 실행 방식): [02-1 · 스크립트 · 1×1 GPU](../../02-script-based/01-single-node-single-gpu/)

## 📚 출처/참고

- 기준 모델 정의: [`00-foundations/recommender-baseline.md`](../../00-foundations/recommender-baseline.md)
- PyTorch DDP/분산 학습 개요: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/
