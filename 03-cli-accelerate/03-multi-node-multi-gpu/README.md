# 03-3 · Accelerate · M×N GPU

> `accelerate launch` 명령 자체를 `TorchDistributor`로 감싸 multi-node DDP로 실행한다. HF Accelerate 인터페이스를 유지하면서 노드 수만 늘리는 패턴.

## 🎯 시나리오

- 03-2에서 검증된 `accelerate launch` 호출을 multi-node로 그대로 확장
- 코드를 Lightning이나 다른 launcher로 재작성하지 않고 노드만 늘리고 싶을 때

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (user 10M × item 5M, emb dim 256) |
| 라이브러리 | `torch`, `accelerate`, `mlflow` |
| 병렬화 | **DDP** (multi-node) |
| 데이터 | parquet shard (UC Volume) |
| 실행 | TorchDistributor가 `accelerate launch` 명령을 각 노드에서 호출 |
| 추적 | MLflow autolog (rank 0) |

## 🖥️ 클러스터 권장 사양

[01-3과 동일](../../01-notebook-only/03-multi-node-multi-gpu/README.md#️-클러스터-권장-사양). Classic 멀티 노드.

## 📂 파일

```
03-multi-node-multi-gpu/
├── README.md
├── driver_notebook.py
├── train.py
└── configs/
    └── accelerate_ddp_multinode.yaml
```

## 🚀 실행 순서

1. driver에서 NUM_NODES·GPUS_PER_NODE·cfg 설정.
2. TorchDistributor가 각 rank에서 `accelerate launch ... train.py`를 호출하도록 wrap.
3. driver에서 등록.

## 🧬 핵심 패턴

```python
import subprocess
from pyspark.ml.torch.distributor import TorchDistributor

def launch_accelerate(cfg_yaml: str, train_args: list[str]):
    cmd = ["accelerate", "launch", "--config_file", cfg_yaml, "train.py", *train_args]
    subprocess.check_call(cmd)

TorchDistributor(
    num_processes=NUM_NODES,    # node 당 1 프로세스 = accelerate launcher
    local_mode=False,
    use_gpu=True,
).run(launch_accelerate, "configs/accelerate_ddp_multinode.yaml", train_args)
```

```yaml
# configs/accelerate_ddp_multinode.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 2
num_processes: 8             # 총 GPU 수 (M*N)
machine_rank: ${MACHINE_RANK}
main_process_ip: ${MASTER_ADDR}
main_process_port: ${MASTER_PORT}
mixed_precision: bf16
```

> `MACHINE_RANK`/`MASTER_ADDR`/`MASTER_PORT`는 TorchDistributor가 환경변수로 주입한다.

## ⚠️ 함정

- TorchDistributor의 `num_processes`는 **노드 수**로 두고, 노드 안의 GPU 수는 accelerate config가 결정한다. 둘이 곱해진 총 GPU 수만 일치하면 된다.
- accelerate config의 `main_process_ip`/`port`를 하드코딩하지 않는다 → 환경변수 치환.

## ➡️ 다음 셀

- 옆: 마지막 열.
- 위: [03-2](../02-single-node-multi-gpu/), [02-3](../../02-script-based/03-multi-node-multi-gpu/)
- 아래: [04-3 · Lightning · M×N GPU](../../04-lightning/03-multi-node-multi-gpu/)

## 📚 출처/참고

- HF Accelerate 분산 학습: https://huggingface.co/docs/accelerate/basic_tutorials/launch
- TorchDistributor: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor
