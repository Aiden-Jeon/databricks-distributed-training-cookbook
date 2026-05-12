# 학습 재개 (Resume)

장시간 분산 학습은 **언제든 죽을 수 있다**는 전제로 짭니다. Databricks 환경에서 죽는 흔한 시나리오:

| 시나리오 | 원인 |
|---------|------|
| Spot interruption | worker 노드가 회수되어 NCCL rendezvous가 깨짐 |
| Cluster auto-termination | inactivity timeout, max runtime 초과 |
| Driver OOM / process death | 메모리 누수, py4j callback 단절 ([torchdistributor-internals.md](torchdistributor-internals.md)) |
| Job retry | Workflow에서 task 실패 후 재시작 |

이 문서는 위 시나리오에서 **마지막 checkpoint부터 학습을 이어받는** 패턴을 정리합니다.

## 무엇을 저장해야 resume이 가능한가

epoch 중간 또는 끝에서 죽었을 때 정확히 같은 위치에서 재개하려면 **model state만으로는 부족**합니다:

| 항목 | 이유 |
|------|------|
| `model.state_dict()` | 학습된 가중치 |
| `optimizer.state_dict()` | AdamW의 momentum/variance — 없으면 lr scheduling이 다시 warmup |
| `lr_scheduler.state_dict()` | scheduler 사용 시 (본 쿡북은 미사용) |
| `epoch` | 어디까지 끝났는지 |
| `global_step` | step 단위 로깅의 x축 연속성 |
| `early_stop.best`, `early_stop.counter` | EarlyStopping 상태 |
| `torch.get_rng_state()` (선택) | 정확한 reproducibility까지 원하면 |

본 쿡북의 trainer는 현재 `{"model": ...}` 만 저장하므로 (`torch_distributor_trainer.py:169`), resume까지 지원하려면 아래 확장이 필요합니다.

## resume-aware 저장 패턴

```python
def save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, early_stop):
    inner = model.module if hasattr(model, "module") else model
    torch.save({
        "model": inner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "early_stop_best": early_stop.best,
        "early_stop_counter": early_stop.counter,
    }, ckpt_path)
```

`uc-volumes-checkpoints.md` 의 `save_checkpoint` 헬퍼는 model만 저장합니다. resume 시 위 시그니처로 확장합니다.

## resume 로직 (학습 함수 시작부)

```python
start_epoch = 0
global_step = 0

# UC Volume의 latest checkpoint가 있으면 로드. 없으면 fresh start.
latest_ckpt = _find_latest(ckpt_dir)        # 아래 유틸 참고
if latest_ckpt and global_rank == 0:
    print(f"resuming from {latest_ckpt}")
if latest_ckpt:
    state = torch.load(latest_ckpt, map_location=device)
    # DDP wrap 전에 state_dict 로드. wrap 후라면 model.module.load_state_dict(...).
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    start_epoch = state["epoch"] + 1
    global_step = state["global_step"]
    early_stop.best = state["early_stop_best"]
    early_stop.counter = state["early_stop_counter"]

for epoch in range(start_epoch, num_epochs):
    ...
```

주의:
- **모든 rank가 같은 checkpoint를 로드해야** 모델이 동기화됩니다. UC Volume은 모든 노드에서 같은 경로로 보이므로 이게 자연스럽게 보장됩니다.
- DDP wrap **전에** load해야 `model.module.load_state_dict()` 분기가 필요 없습니다.
- `map_location=device` 로 rank별로 GPU 메모리에 직접 올려서 driver memory 우회.

### latest checkpoint 찾기

```python
import glob

def _find_latest(ckpt_dir):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint-*.pt")))
    return paths[-1] if paths else None
```

UC Volume에 epoch 단위로 저장하면 `checkpoint-0.pt`, `checkpoint-1.pt`, ... 식으로 쌓이고 lexical sort가 그대로 epoch 순서가 됩니다 (단, 10 epoch 이상 가면 `checkpoint-09.pt` 처럼 zero-pad 필요).

## MLflow run reattach

`mlflow.start_run(run_id=...)` 으로 이전 run에 이어 metric을 append할 수 있습니다. 단:

- **새로운 run을 만들 것인가, 이전 run에 이어붙일 것인가**는 정책 선택입니다. 권장:
  - **같은 학습 시도의 재시작** (예: spot interruption) → 같은 `run_id` 에 attach. timeline이 끊기지 않습니다.
  - **다른 시도** (예: hyperparameter 변경) → 새 run.
- run_id를 어디에 보존할지: `ckpt_dir` 옆에 `_run.json` 으로 저장하는 것이 단순합니다.

```python
# driver 셀에서
run_meta_path = os.path.join(CKPT_DIR, "_run.json")
if os.path.exists(run_meta_path):
    with open(run_meta_path) as f:
        run_id = json.load(f)["run_id"]
    print(f"reattaching to run {run_id}")
else:
    with mlflow.start_run(run_name="recommender-MxN", log_system_metrics=True) as run:
        run_id = run.info.run_id
    with open(run_meta_path, "w") as f:
        json.dump({"run_id": run_id}, f)

TorchDistributor(...).run(train_fn_ddp, run_id=run_id, ...)
```

resume된 step의 x축 충돌: `mlflow.log_metric("train/loss", ..., step=global_step)` 에서 `global_step` 을 checkpoint에서 이어받으면 자연스럽게 연속됩니다. 같은 step에 두 번 log 하면 MLflow는 두 값을 모두 보존합니다 (UI는 timestamp 순으로 그림).

## Job retry와 결합

Databricks Workflow의 task가 실패하면 `max_retries` 만큼 재실행됩니다. **학습 함수가 idempotent하게 resume을 처리하면** retry가 자동 resume이 됩니다.

핵심 조건:
1. `ckpt_dir` 가 retry 간 보존되는 경로 (UC Volume — `/local_disk0`은 클러스터 종료 시 소실)
2. `run_id` 가 retry 간 보존 (위 `_run.json` 패턴)
3. 학습 함수가 시작 시 `_find_latest` 로 자동 복귀

`02-script-based/08-launch_accelerator_MxN.ipynb` 의 `dbutils.notebook.exit(f"... rc={p.returncode}")` 가 non-zero exit으로 retry를 트리거합니다 — 이 패턴과 결합하면 spot interruption → notebook fail → retry → resume 흐름이 자동 됩니다.

## 저장 빈도

| 빈도 | 장점 | 단점 |
|------|------|------|
| epoch마다 | 단순, MLflow `LoggedModel` 과 일치 | 긴 epoch에서 손실 윈도우 큼 |
| step `% 500 == 0` | spot 회수에 강함 | I/O 부하, optimizer state는 크므로 UC Volume write 비쌈 |
| 둘 다 (step은 local, epoch은 UC) | hybrid | 복잡 |

본 쿡북 권장: **epoch마다 UC Volume에 직접 저장**. ML-25M + `max_steps_per_epoch=200` 기준 epoch이 짧아 손실 윈도우가 작습니다.

## 함정

- **DDP wrap 후 `load_state_dict`**: wrap된 모델의 key는 `module.user_emb.weight` 처럼 prefix가 붙습니다. wrap 전에 load하거나 `model.module.load_state_dict(state)` 로 호출.
- **optimizer state device mismatch**: `torch.load(..., map_location=device)` 후에도 optimizer state의 tensor들은 따로 `.to(device)` 가 필요할 수 있습니다. `optimizer.load_state_dict(state["optimizer"])` 호출 후 첫 `optimizer.step()` 에서 PyTorch가 자동 처리하지만, rank별 device가 다르면 명시 이동 필요.
- **early stopping double-counting**: `early_stop.counter` 를 복원하지 않으면 patience가 리셋되어 학습이 무한정 길어질 수 있습니다.
- **resume + autoscaling**: autoscaling으로 노드 수가 바뀌면 `WORLD_SIZE` 가 달라져 학습 dynamics가 변합니다. 본 쿡북은 autoscaling OFF 가정.

## 참고

- PyTorch checkpoint 가이드: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
- MLflow run resume: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
- Databricks Job retries: https://docs.databricks.com/aws/en/jobs/repair-job-failures
- 본 쿡북의 checkpoint 저장: [`uc-volumes-checkpoints.md`](uc-volumes-checkpoints.md)
