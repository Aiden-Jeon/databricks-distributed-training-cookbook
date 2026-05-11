# 99 · References

본 쿡북에서 인용·참고한 외부 자료 인덱스. 새 자료는 여기 먼저 추가하고, 셀 README의 "📚 출처/참고" 섹션에서 링크합니다.

## 📂 인덱스

| 문서 | 내용 |
|------|------|
| [official-notebooks-mirror.md](official-notebooks-mirror.md) | Databricks 공식 분산 학습 예제 노트북 (LLM 중심, 본 쿡북 패턴과 비교용) |

> 본 쿡북은 **추천 모델(Two-Tower MLP)** 시나리오로 작성되었지만, Databricks 공식 분산 학습 예제는 대부분 LLM fine-tuning입니다. 공식 예제는 launcher·토폴로지 패턴을 참고하는 용도로 인덱싱합니다.

## 🔗 메인 진입점

- 분산 학습 개요: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/
- Multi-GPU workload (`@distributed`): https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training
- Multi-GPU 예제 인덱스: https://docs.databricks.com/aws/en/machine-learning/sgc-examples/gpu-distributed-training
- TorchDistributor: https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor

## 🤗 HuggingFace

- HF Accelerate: https://huggingface.co/docs/accelerate/index
- HF Accelerate 분산 학습: https://huggingface.co/docs/accelerate/basic_tutorials/launch

## ⚡ Lightning

- PyTorch Lightning Trainer: https://lightning.ai/docs/pytorch/stable/common/trainer.html
- Lightning DDP: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
- Lightning multi-node: https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_1.html

## 📦 데이터 로딩

- Prepare data for distributed training: https://docs.databricks.com/aws/en/machine-learning/load-data/ddl-data
- `torch.utils.data.distributed.DistributedSampler`: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
