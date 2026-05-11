# Snippets · 내부 참조 코드

이전 진행 프로젝트에서 가져온 **포팅 원본**. 본 쿡북의 12개 셀과 `00-foundations/` 문서로 옮길 때 참고하는 ground truth.

> 이 폴더의 파일은 **원본 그대로**다. 본 쿡북의 도메인(추천)이나 모델(Two-Tower MLP)에 맞춰 수정되지 않았습니다. 셀 코드를 채울 때는 여기서 패턴만 가져오고 모델·하이퍼파라미터는 셀별 스케일에 맞춥니다.

## 📂 출처

| 디렉터리 | 원본 경로 | 도메인 |
|---------|----------|--------|
| `torch_distributed/` | `archives/torch_distributed/` (2025-03) | HuggingFace DistilBERT, IMDB 감성 분류 |
| `fashion_recommendations/` | `projects/2601-musinsa/fashion_recommendations/` (2026-01) | H&M 패션 추천, LSTM sequential model |

## 🧭 패턴 → 본 쿡북 셀 매핑

### `torch_distributed/`

| 파일 | 핵심 패턴 | 매칭되는 셀/문서 |
|------|----------|-----------------|
| `02-torch_distributor.py` | `TorchDistributor(num_processes=N, local_mode=False, use_gpu=True).run(train_fn, ...)` + `dist.init_process_group("nccl")` + `DATABRICKS_HOST/TOKEN`을 driver→worker로 전달 | **01-3, 02-3** (M×N GPU) |
| `01-simple_huggingface_train.py` | 단일 GPU `train_single_node` 함수 + `mlflow.start_run` 골격 | **01-1, 02-1** (1×1 GPU) |
| `src/train.py` | `train_one_epoch(model, optimizer, loader, device, epoch, log_interval, max_duration)` + `mlflow.log_metric` | 모든 셀 학습 루프 |
| `src/eval.py` | `AverageMeter.all_reduce(device)` — `dist.all_reduce(ReduceOp.SUM)`로 rank 간 metric 합산 | 모든 multi-GPU 셀 eval |
| `src/utils.py` | `create_log_dir(volume_dir)` (timestamp suffix), `save_checkpoint(log_dir, model, epoch)` (state_dict 저장) | `00-foundations/uc-volumes-checkpoints.md` |
| `src/model.py` | 모델 정의 패턴 (HF backbone + classifier head). **모델 자체는 사용 안 함** — Two-Tower MLP로 교체 | 참고용 |
| `src/dataset.py` | HF `load_dataset` + tokenizer + `set_format("torch", ...)`. **HF 데이터는 사용 안 함** — synthetic interaction Delta로 교체. MDS/Streaming 부분도 사용 안 함 | 참고용 |

### `fashion_recommendations/`

| 파일 | 핵심 패턴 | 매칭되는 셀/문서 |
|------|----------|-----------------|
| `pytorch_utils.py` `LSTMTrainerDistributed` | DDP 학습 루프 (`sampler.set_epoch(epoch)`, rank-0 로깅, `clip_grad_norm_`, early stopping) — 모델만 교체하면 그대로 재사용 | **01-2, 01-3, 02-2, 02-3** |
| `pytorch_utils.py` `LSTMRecommenderLightning` | `LightningModule` (`training_step`, `validation_step`, `sync_dist=True`, `save_hyperparameters()`, `configure_optimizers`) | **04-1, 04-2, 04-3** |
| `pytorch_utils.py` `LSTMTrainer` | non-distributed 학습 루프 (early stopping + checkpoint save/load) | **01-1, 02-1** |
| `train_simple_mlp.ipynb` | MLflow 3.0+ 종합 패턴: `log_system_metrics=True`, 단일 run에서 epoch별 `mlflow.pytorch.log_model(..., step=epoch, name=...)`, `mlflow.data.from_pandas`, `mlflow.log_metric(..., model_id=, dataset=)`, UC 모델 등록, `client.get_logged_model`로 best 검색 | `00-foundations/mlflow-tracking.md` + **01-1** |
| `data_utils.py` | `load_delta_table(name)` — `spark.table()` 래퍼. `create_ground_truth_labels` 등 평가 헬퍼 | `00-foundations/data-loading.md` |
| `model_config.py` | 모델별 하이퍼파라미터 dict (`POPULARITY_CONFIG`, `ENSEMBLE_CONFIG` 형식). LSTM은 본 쿡북에서 사용 안 함 | 셀별 config 파일의 형식 참고용 |

## ⚠️ 스코프 밖 (가져오지 않음)

원본에 있었으나 본 쿡북에서 사용하지 않는 것:

| 원본 파일/패턴 | 제외 이유 |
|---------------|----------|
| `torch_distributed/04-train_with_composer.py`, `examples/_distributed_composer/*` | MosaicML Composer — 본 쿡북 비대상 |
| `torch_distributed/03-ray_train.py`, `examples/ray/*` | Ray Train — 본 쿡북 비대상 (TorchDistributor만 다룸) |
| `dataset.py` 의 `load_mds_dataset`, `get_mds_dataloader` | MDS streaming — 본 쿡북은 parquet shard만 다룸 |
| `fashion_recommendations/mlops/training/notebooks/train_lstm.ipynb` 등 | LSTM, popularity, age rules, ensemble 등 모델 자체 — Two-Tower MLP만 다룸 |
| `pytorch_utils.py` `LSTMRecommender`, `generate_recommendations` | LSTM 모델 정의·inference — 모델만 교체 |

## 📜 라이선스 / 출처 표기

두 원본 모두 Databricks Field Engineering 내부 프로젝트. 본 쿡북도 같은 컨텍스트이므로 코드 이동은 자유롭습니다. 셀 README의 "📚 출처/참고"에는 패턴을 가져온 원본 파일을 명시합니다 (예: "DDP 학습 루프는 `fashion_recommendations/mlops/utils/pytorch_utils.py:LSTMTrainerDistributed` 기반").

## ➡️ 다음 단계

이 snippet들을 다음 순서로 본 쿡북 셀에 포팅합니다:

1. `00-foundations/`의 utility 모듈 (모델 정의, AverageMeter, checkpoint, MLflow 헬퍼) 정리
2. **01-1** (1×1 GPU 노트북) — `train_simple_mlp.ipynb` + `train_single_node` 합쳐서 baseline
3. **01-2** (1×N GPU 노트북) — `LSTMTrainerDistributed` + `@distributed` 데코레이터
4. **01-3** (M×N GPU 노트북) — `02-torch_distributor.py` 패턴 적용
5. **02-1/2/3** (스크립트) — 01 시리즈를 `train.py` + `driver_notebook.py`로 분리
6. **03-1/2/3** (Accelerate) — `train.py`를 `accelerate launch` 진입점으로 wrap
7. **04-1/2/3** (Lightning) — `LSTMRecommenderLightning` 패턴 적용
