# UC Volumes에 체크포인트 저장

분산 학습에서 체크포인트를 어디에 두느냐는 안정성에 직결된다. 본 쿡북의 모든 셀은 **Unity Catalog Volumes**를 사용한다.

## 왜 UC Volume인가

| 후보 | 장점 | 단점 |
|------|------|------|
| `/local_disk0/...` | 가장 빠름 | 클러스터 종료와 함께 소실 |
| DBFS | 모든 노드 공유 | 레거시, 권한 모델 불편 |
| **UC Volume** | UC 권한·lineage, 모든 노드 공유, 클라우드 객체 스토리지 | I/O 지연 (HDD 수준) |

→ 큰 체크포인트는 일단 `/local_disk0/`에 쓰고 학습 종료 후 UC Volume으로 복사하는 hybrid가 일반적.

## 패턴 1: 학습 중 local disk + 종료 후 UC Volume copy

```python
local_ckpt = "/local_disk0/ckpt/run-001"
uc_ckpt = "/Volumes/main/recsys/checkpoints/run-001"

# 학습 루프: rank 0만 주기적으로 local disk에 저장
if rank == 0 and step % save_steps == 0:
    torch.save(model.state_dict(), f"{local_ckpt}/step-{step}.pt")

# 학습 종료 후 최종만 UC Volume으로 복사
if rank == 0:
    shutil.copytree(local_ckpt, uc_ckpt, dirs_exist_ok=True)
```

## 패턴 2: 직접 UC Volume에 저장

체크포인트 경로를 바로 `/Volumes/...`로 지정. 단일 노드·작은 모델에서만 권장. Multi-node에서 모든 rank가 동시에 쓰면 I/O 병목이 생긴다.

## Multi-node 주의

- **모든 노드가 같은 경로를 본다.** `/Volumes/...`는 클러스터 전체에 마운트됨.
- 그러나 동시에 쓰면 contention 발생 → save 주기를 크게 두거나 rank 0만 저장.
- DDP는 모델이 모든 rank에 동일하게 복제되므로 **rank 0의 state_dict만 저장**하면 충분하다.

## 디렉토리 컨벤션 (본 쿡북)

```
/Volumes/<catalog>/<schema>/checkpoints/<run_name>/
    └── final.pt
/Volumes/<catalog>/<schema>/datasets/<dataset_name>/...
```

## 헬퍼 함수

각 셀에서 그대로 복사해서 쓰는 헬퍼. UC Volume 경로 끝에 타임스탬프 디렉터리를 만들어 run별 격리를 보장하고, `model.state_dict()`만 저장한다 (DDP-wrapped 모델은 inner module을 꺼낸다).

```python
import os
import time
import torch


def create_log_dir(log_volume_dir: str) -> str:
    """UC Volume 아래에 타임스탬프 디렉토리를 만들고 경로를 반환."""
    log_dir = os.path.join(log_volume_dir, str(time.time()))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_checkpoint(log_dir: str, model: torch.nn.Module, epoch: int) -> str:
    """rank 0에서만 호출. DDP/Lightning 등 래퍼를 벗긴 뒤 state_dict를 저장."""
    inner = model.module if hasattr(model, "module") else model
    filepath = os.path.join(log_dir, f"checkpoint-{epoch}.pt")
    torch.save({"model": inner.state_dict(), "epoch": epoch}, filepath)
    return filepath
```

분산 학습 컨텍스트:

```python
import os

rank = int(os.environ.get("RANK", "0"))
if rank == 0:
    save_checkpoint(log_dir, model, epoch)

# 모든 rank가 다음 epoch을 동기적으로 시작하도록 barrier
import torch.distributed as dist
if dist.is_initialized():
    dist.barrier()
```

## 참고

- 데이터/체크포인트 디렉토리 구조는 셀별 README의 `00_setup.py`에서 변수로 받게 한다.
- 코드 출처: [`99-references/snippets/torch_distributed/src/utils.py`](../99-references/snippets/torch_distributed/src/utils.py)
