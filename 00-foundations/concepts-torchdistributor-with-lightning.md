# TorchDistributor × Lightning 조합

TorchDistributor는 launcher고 PyTorch Lightning은 training framework입니다. 둘은 같은 층에 있는 도구가 아니라 위·아래로 겹쳐 쓰는 도구이므로, 어떻게 조합하느냐에 따라 동작이 크게 달라집니다. 이 문서는 본 쿡북의 Lightning 셀이 어떤 조합을 택했는지, 그리고 잘못 조합했을 때 어떤 일이 벌어지는지 정리합니다.

## Lightning에는 왜 `local_mode`가 없는가 — driver는 어떻게 쓰이는가

같은 분산 학습이라도 Lightning에는 `local_mode`에 해당하는 옵션이 보이지 않습니다. 이유는 단순합니다. `local_mode`는 TorchDistributor의 **launcher 옵션**인 반면, PyTorch Lightning은 launcher가 아니라 training framework이기 때문입니다. Lightning은 분산 환경 자체는 외부 launcher(`torchrun`, SLURM, TorchDistributor 등)가 잡아 준다는 전제 위에서 동작합니다.

> "Lightning implementation of DDP calls your script under the hood multiple times with the correct environment variables." / "simply launch your script with the torchrun command." — [PyTorch Lightning Docs: GPU training (Intermediate)](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html)

그래서 `L.Trainer`가 받는 인자도 `devices`, `num_nodes`, `strategy` 정도로 단순합니다. "어디(driver냐 worker냐)에서 프로세스를 띄울지"는 Trainer가 알 필요가 없고, 그것은 한 단계 위에서 프로세스를 띄우는 launcher의 몫입니다.

결과적으로 driver를 학습에 쓰느냐 마느냐는 **launcher가 어떻게 잡아 주느냐**에 따라 갈립니다.

| 시나리오 | 프로세스를 띄우는 주체 | driver 사용 |
|---------|--------------------|------------|
| 노트북에서 `Trainer.fit()` 직접 호출 (single-node) | Lightning이 노트북 프로세스 자체에서 child 1~N개 spawn | O — driver의 GPU 사용 |
| `TorchDistributor(local_mode=True)` + Lightning | TorchDistributor가 driver 안에서 N개 프로세스 spawn, 각자가 `Trainer.fit()` 호출 | O |
| `TorchDistributor(local_mode=False)` + Lightning | Spark가 worker 노드들에 분산 spawn | X — driver는 코디네이션만 |
| 노트북에서 `Trainer(num_nodes=M>1)` 직접 호출 | Lightning은 worker 노드에 프로세스를 띄울 수단이 없음 | **실패** |

## 본 쿡북의 셀 매핑

본 쿡북의 Lightning 셀(`01-notebook-based/04-train_pytorch_lightning_1x1_1xN.ipynb`, `05-train_pytorch_lightning_MxN.ipynb`)은 위 표 중 1·2·3행에 차례로 매핑됩니다.

| 셀 | launcher | Trainer 호출 |
|----|----------|-------------|
| Lightning 1×1 | 없음 — 노트북에서 직접 | `Trainer(devices=1, num_nodes=1).fit(...)` |
| Lightning 1×N | `TorchDistributor(num_processes=N, local_mode=True)` | `Trainer(devices=N, num_nodes=1, strategy="ddp").fit(...)` |
| Lightning M×N | `TorchDistributor(num_processes=M*N, local_mode=False)` | `Trainer(devices=N, num_nodes=M, strategy="ddp").fit(...)` |

1×1만 TorchDistributor 없이 직접 호출하는 데는 이유가 있습니다. Lightning 단독으로도 single-process 학습이 완결되기 때문에, launcher를 한 겹 더 두는 가치가 거의 없습니다. 반면 1×N부터는 TorchDistributor가 프로세스 스폰과 NCCL 환경 변수(`RANK`, `WORLD_SIZE`, `MASTER_ADDR` 등)를 일관되게 잡아 주는 이점이 분명하므로 감싸 줍니다. M×N은 worker 노드에 프로세스를 분배해 줄 주체가 필요하므로 **반드시** `local_mode=False`로 감싸야 합니다 ([Databricks Blog: PyTorch on Databricks – Introducing the Spark PyTorch Distributor](https://www.databricks.com/blog/2023/04/20/pytorch-databricks-introducing-spark-pytorch-distributor.html)).

> Databricks 공식 블로그는 multi-node Lightning에서 `Trainer(devices=1, num_nodes=num_processes)`도 권장 패턴으로 소개합니다. 각 TorchDistributor 프로세스를 독립된 "node"로 취급해 Lightning이 추가 child를 spawn하지 않게 만드는 단순화 패턴입니다. 본 쿡북은 클러스터 토폴로지를 그대로 드러내려는 의도로 `devices=N, num_nodes=M`을 사용합니다. 둘 다 동작하지만 의미와 디버깅 관점이 다르다는 점만 알아 두면 됩니다.

## 함정

Lightning을 TorchDistributor와 조합할 때 자주 만나는 함정도 정리해 둡니다.

- 노트북에서 `Trainer(num_nodes=M>1)`를 TorchDistributor 없이 직접 호출하면 Lightning이 worker 노드에 접근할 수단이 없어 그대로 멈춥니다. Databricks 환경에서 multi-node Lightning은 반드시 `TorchDistributor(local_mode=False)`로 감싸야 합니다.
- 반대로 `TorchDistributor(local_mode=True, num_processes=N)` 안쪽에서 `Trainer(num_nodes>1)`을 주는 조합도 피해야 합니다. TorchDistributor는 driver 한 대 안에서 N개 프로세스를 띄우는데, Lightning에는 "여러 노드"라고 거짓말을 하는 셈이 되어 rendezvous가 어긋납니다.
- TorchDistributor의 child 프로세스는 노트북과 별개인 fresh Python에서 시작합니다. 그래서 LightningModule, DataModule, callback 같은 객체는 `train_fn` 안에서 다시 정의하거나 import해 주어야 합니다 ([`debug-common-pitfalls.md §2`](../docs/debug-common-pitfalls.md)).
