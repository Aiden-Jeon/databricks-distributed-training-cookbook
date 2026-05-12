# Official Notebooks Mirror

Databricks 공식 분산 학습 예제 노트북을 모아 둔 미러입니다. 본 쿡북의 12개 셀은 launcher·토폴로지 패턴을 일반화한 것이며, 모델 도메인만 **추천(Two-Tower MLP)**으로 바꿔 사용합니다. 공식 예제는 패턴 검증과 디버깅 시 원본 참조용으로 활용하세요.

## 🧱 NLP fine-tuning 공식 예제 (참고용)

공식 예제를 모델·토폴로지·병렬화 축으로 정리하면 다음과 같습니다.

| 모델 | 토폴로지 | 병렬화 | 링크 |
|------|---------|--------|------|
| GPT-OSS 20B | 1×N GPU | DDP | [sgc-distributed-gpt-oss-20b](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-distributed-gpt-oss-20b) |
| GPT-OSS 120B | M×N GPU | FSDP | [sgc-gpt-oss-120b-ddp-fsdp](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-gpt-oss-120b-ddp-fsdp) |
| Llama 3.2 1B | 1×1 GPU | SFT (TRL) | [sgc-sft-trl-deepspeed-llama-1b](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-sft-trl-deepspeed-llama-1b) |
| Llama 3.2 3B | 1×N GPU | Unsloth distributed | [sgc-finetune-llama-unsloth-distributed](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-llama-unsloth-distributed) |
| Llama 3 8B | M×N GPU | LLM Foundry | [sgc-llama3-8b-llmfoundry](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-llama3-8b-llmfoundry) |
| Olmo3 7B | M×N GPU | Axolotl + LoRA | [sgc-olmo3-7b-lora-axolotl](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-olmo3-7b-lora-axolotl) |

## 🧭 본 쿡북 셀과의 launcher/토폴로지 대응

LLM 예제이지만 launcher·토폴로지 패턴은 동일합니다. 본 쿡북에서는 모델만 Two-Tower MLP로 교체했습니다.

| 본 쿡북 셀 | 대응 공식 예제의 launcher/토폴로지 |
|-----------|-----------------------------------|
| 01-1 (NB · 1×1) | Llama 3.2 1B (단일 GPU 학습 루프) |
| 01-2 (NB · 1×N) | GPT-OSS 20B DDP, Llama 3.2 3B Unsloth (`@distributed`) |
| 01-3 (NB · M×N) | GPT-OSS 120B (TorchDistributor multi-node) |
| 02-3 / 03-3 (Script/Accelerate · M×N) | Llama 3 8B (LLM Foundry), Olmo3 7B (Axolotl) |

> 공식 예제는 FSDP를 자주 사용하지만, 본 쿡북의 모델은 단일 GPU에 들어가므로 FSDP 부분은 무시하고 DDP 패턴만 참고합니다.

## 📝 인용 규칙

셀 README의 "📚 출처/참고" 섹션에는 본 쿡북 도메인에 맞는 자료(추천·DDP·PyTorch·Lightning 문서)를 인용합니다. 위의 LLM 공식 예제는 launcher 사용법을 디버깅할 때만 참조하세요.
