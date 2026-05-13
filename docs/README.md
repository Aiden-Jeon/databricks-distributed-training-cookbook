# 참고 자료

본 쿡북이 인용·참고하는 외부 자료, 공식 노트북, 그리고 작업 초기 메모를 모아 둔 곳입니다. 새 자료는 이곳에 먼저 추가한 뒤, 셀 README의 "📚 출처/참고" 섹션에서 링크합니다.

## 📂 인덱스

문서별 위치는 다음과 같습니다.

| 문서 | 내용 |
|------|------|
| [official-notebooks-mirror.md](official-notebooks-mirror.md) | Databricks 공식 분산 학습 예제 노트북 (LLM 중심, 본 쿡북 패턴과 비교용) |
| [debug-common-pitfalls.md](debug-common-pitfalls.md) | pickle, NCCL, OOM 등 자주 부딪히는 함정 + 진단 표 |

> 본 쿡북은 **추천 모델(Two-Tower MLP)** 시나리오로 작성되었지만, Databricks 공식 분산 학습 예제는 대부분 LLM fine-tuning입니다. 공식 예제는 launcher·토폴로지 패턴을 참고하는 용도로 인덱싱합니다.

## 🔗 메인 진입점

자세한 내용은 다음 자료를 참조하세요.

- [Databricks 분산 학습 개요](https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/)
- [Multi-GPU workload (`@distributed`)](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training)
- [Multi-GPU 예제 인덱스](https://docs.databricks.com/aws/en/machine-learning/sgc-examples/gpu-distributed-training)
- [TorchDistributor 문서](https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor)

## 🤗 HuggingFace

자세한 내용은 다음 자료를 참조하세요.

- [HF Accelerate 개요](https://huggingface.co/docs/accelerate/index)
- [HF Accelerate 분산 학습 튜토리얼](https://huggingface.co/docs/accelerate/basic_tutorials/launch)

## ⚡ Lightning

자세한 내용은 다음 자료를 참조하세요.

- [PyTorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
- [Lightning DDP](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html)
- [Lightning multi-node](https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_1.html)

## 📦 데이터 로딩

자세한 내용은 다음 자료를 참조하세요.

- [Prepare data for distributed training](https://docs.databricks.com/aws/en/machine-learning/load-data/ddl-data)
- [`torch.utils.data.distributed.DistributedSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
