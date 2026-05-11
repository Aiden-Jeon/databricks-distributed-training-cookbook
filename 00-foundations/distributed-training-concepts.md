# Distributed Training Concepts

본 쿡북에서 반복적으로 등장하는 핵심 개념을 한 문서에 모은다.

## 본 쿡북이 다루는 병렬화: DDP 한 가지

| 항목 | DDP (DistributedDataParallel) |
|------|-------------------------------|
| 모델 메모리 | 각 GPU에 모델 **전체** 복제 |
| 그래디언트 | 각 GPU 계산 후 AllReduce |
| 데이터 | rank별로 미니배치를 나눠 처리 |
| 통신량 | 작음 (AllReduce 1회/스텝) |
| 적합 모델 크기 | 한 GPU에 모델 + optimizer state가 들어가는 크기 |

본 쿡북의 기준 모델([recommender-baseline.md](recommender-baseline.md))은 가장 큰 셀에서도 수 GB 임베딩 테이블 + 작은 MLP라 단일 GPU에 들어간다. 따라서 **DDP만으로 충분**하다. FSDP(모델 샤딩)는 다루지 않는다.

> 토폴로지(1×1 / 1×N / M×N)는 **GPU 개수**의 문제고, DDP는 **여러 GPU에 같은 모델을 복제해 데이터 병렬 처리**하는 *전략*이다. 1×N과 M×N 모두 DDP를 쓰며, 차이는 GPU가 노드 안에만 있는지 노드를 가로지르는지뿐이다.

## 왜 multi-node DDP인가 (M×N 셀의 정당화)

작은 MLP에서 multi-node DDP는 두 가지 이유로 의미가 있다:

1. **데이터 처리량 확장.** 수십억 행 user-item interaction을 노드 수에 비례하게 처리할 수 있다.
2. **TorchDistributor 사용법 시연.** Databricks에서 multi-node 분산 launcher를 어떻게 띄우는지를 같은 모델 골격으로 보여주기 위함.

> 다만 모델이 작을수록 AllReduce 통신 비용이 상대적으로 커진다. 노드를 무작정 늘리면 throughput이 비례하지 않을 수 있다. 본 쿡북의 M×N 셀들은 "패턴이 동작한다"는 것을 보이는 데 초점을 두며, 본격 스케일링 실험은 사용자 워크로드에 맞춰 별도로 진행한다.

## Mixed precision

| 모드 | 설명 | 사용 시점 |
|------|------|----------|
| FP32 | 기준점, 메모리 4×param | 디버깅용 |
| FP16 + GradScaler | 절반 메모리, overflow 위험 | 구형 GPU(V100) |
| BF16 | 절반 메모리, scaler 불필요 | Ampere(A100)/Hopper(H100). **기본 선택**. |

본 쿡북의 셀은 모두 BF16을 가정한다. 모델이 작아 mixed precision 효과가 LLM만큼 극적이진 않지만, AdamW 옵티마이저 상태(파라미터의 2배)까지 고려하면 메모리 절감은 여전히 의미가 있다.

## 임베딩 테이블이 큰 경우

추천 모델에서 파라미터 대부분은 `nn.Embedding`에서 나온다. DDP는 임베딩 테이블도 GPU마다 복제하므로 다음을 염두에 둔다:

- 임베딩 테이블이 GPU 메모리의 큰 비중을 차지하면 `per_device_batch_size`를 충분히 줄여야 한다.
- `optimizer.zero_grad(set_to_none=True)`로 그래디언트 메모리를 회수한다.
- 정말 거대한 임베딩(수십~수백 GB)이라면 `TorchRec`처럼 임베딩을 GPU 간 샤딩하는 라이브러리가 필요하지만 본 쿡북 범위 밖이다.

## 학습률·배치·grad accumulation

- 분산 학습은 **effective batch size = per_device_batch × num_gpus × grad_accum_steps**.
- 학습률은 effective batch가 커질수록 일반적으로 비례 증가가 필요하다 (linear scaling rule).
- PyTorch 학습 루프에서는 위 값을 직접 곱해 lr을 조정한다. Lightning `Trainer`나 HF `TrainingArguments`는 인자만 받는다.

## 참고

- PyTorch DDP 가이드: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- Databricks 분산 학습 개요: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/
