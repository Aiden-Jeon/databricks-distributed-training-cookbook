# 02-1 · 스크립트 · 1×1 GPU

> `train.py`로 학습 로직을 파일에 분리하고, 노트북은 driver 역할만 한다. 1×1 GPU에서는 가장 단순한 호출 방식.

## 🎯 시나리오

- 같은 학습 코드를 여러 데이터셋·하이퍼파라미터로 반복 실행
- 노트북보다 git diff·코드 리뷰가 중요한 단계
- 추후 1×N, M×N으로 확장할 때 코드를 재사용하고 싶을 때

## 🧱 스택

| 항목 | 선택 |
|------|------|
| 모델 | Two-Tower MLP (01-1과 동일, [recommender-baseline](../../00-foundations/recommender-baseline.md)) |
| 라이브러리 | `torch`, `mlflow` |
| 병렬화 | 단일 GPU |
| 데이터 | 합성 interaction → Delta → DataLoader |
| 실행 | `%run ./train.py` 또는 `!python train.py --config configs/training_args.yaml` |
| 추적 | MLflow autolog |

## 🖥️ 클러스터 권장 사양

[01-1과 동일](../../01-notebook-only/01-single-node-single-gpu/README.md#️-클러스터-권장-사양).

## 📂 파일

```
01-single-node-single-gpu/
├── README.md
├── driver_notebook.py       # 데이터 준비 + train.py 호출 + 등록
├── train.py                 # 학습 entrypoint (argparse / config)
└── configs/
    └── training_args.yaml   # 모델·데이터·하이퍼파라미터
```

## 🚀 실행 순서

1. `driver_notebook.py`를 노트북으로 import.
2. `configs/training_args.yaml`을 시나리오에 맞게 수정.
3. driver 셀에서 `!python train.py --config configs/training_args.yaml` 호출.
4. driver에서 평가·UC 등록.

## 🧬 핵심 패턴

```python
# train.py
def main(cfg):
    import mlflow, torch
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=cfg["run_name"]):
        model = TwoTowerMLP(cfg["n_users"], cfg["n_items"], cfg["emb_dim"]).cuda()
        # ... 학습 루프 ...

if __name__ == "__main__":
    main(load_config())
```

```python
# driver_notebook.py
%sh python train.py --config configs/training_args.yaml
```

## ⚠️ 함정

- `!python`은 driver의 Python 환경을 그대로 쓴다. 별도 venv가 필요하면 `%pip install`을 먼저.
- driver 셀에서 `mlflow.start_run`을 또 열면 nested run이 되니, run은 `train.py`에서만 시작한다.

## ➡️ 다음 셀

- 옆: [02-2 · 스크립트 · 1×N GPU](../02-single-node-multi-gpu/)
- 위 (다시 노트북 only로): [01-1](../../01-notebook-only/01-single-node-single-gpu/)
- 아래 (Accelerate로 launcher 교체): [03-1](../../03-cli-accelerate/01-single-node-single-gpu/)

## 📚 출처/참고

- 기준 모델 정의: [`00-foundations/recommender-baseline.md`](../../00-foundations/recommender-baseline.md)
