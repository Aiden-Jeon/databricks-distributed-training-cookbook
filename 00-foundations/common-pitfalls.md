# 흔한 함정

분산 학습에서 자주 부딪히는 실패와 해결법. 디버깅 중이면 여기부터 본다.

## 1. NCCL 초기화 실패 / 타임아웃

증상: `NCCL error ... unhandled cuda error` 또는 첫 스텝에서 멈춤.

원인:
- 노드 간 통신 포트가 막힘.
- `MASTER_ADDR`/`MASTER_PORT`가 잘못됨.
- IB/EFA가 활성화되지 않아 default ethernet으로 fallback.

해결:
- TorchDistributor를 쓰면 위 환경변수는 자동 설정됨 → 직접 만들지 말 것.
- 로그에 `NCCL INFO` 라인을 보고 어떤 transport를 쓰는지 확인.
- `NCCL_DEBUG=INFO`, `NCCL_SOCKET_IFNAME=eth0` 등 조정.

## 2. pickle 에러 (TorchDistributor.run(fn, ...))

증상: `Can't pickle local object`, `cannot pickle '_thread.RLock'`.

원인: `fn`이 closure로 driver 변수(SparkSession, MLflow client 등)를 캡처.

해결:
- 학습 함수는 **모듈 최상위 또는 별도 .py 파일에 정의**.
- 인자는 primitives(str, int, list)만 넘긴다.
- SparkSession은 함수 내부에서 `SparkSession.builder.getOrCreate()`로 다시 얻는다.

## 2-1. child 프로세스에서 MLflow가 인증 실패

증상: `mlflow.start_run(run_id=...)`이 `401 Unauthorized` 또는 `Could not find experiment`.

원인: TorchDistributor가 띄운 worker 프로세스는 driver의 Databricks 자격증명을 자동 상속하지 않는다.

해결: driver에서 host/token을 명시적으로 읽어 학습 함수의 인자로 넘기고, child 안에서 환경변수에 다시 설정한다.

```python
# Driver cell
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
db_host = context.extraContext().apply("api_url")
db_token = context.apiToken().get()

TorchDistributor(...).run(train_fn, db_host=db_host, db_token=db_token, ...)
```

```python
def train_fn(db_host, db_token, run_id, ...):
    import os
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token
    # 이제 mlflow가 자격증명을 찾을 수 있다
```

## 3. OOM at first forward

증상: 첫 forward에서 즉시 OOM.

원인:
- `per_device_batch_size`가 너무 큼.
- 임베딩 테이블이 GPU 메모리에서 큰 비중을 차지하는데 배치 크기를 함께 키움.
- AdamW 옵티마이저 상태(파라미터의 2배)를 잊고 추산.

해결:
1. `per_device_batch_size`를 절반으로 줄이고 점진적으로 늘린다.
2. `optimizer.zero_grad(set_to_none=True)`로 그래디언트 메모리 회수.
3. 임베딩이 너무 크면 `cluster-recipes.md`에서 더 큰 GPU로 이동.

## 4. DDP에서 학습이 멈춘 것처럼 보인다

원인:
- rank 간 데이터 분할이 불균등해 누군가 먼저 끝나면 다른 rank가 AllReduce를 기다린다.
- 모델에 사용되지 않은 파라미터가 있으면 DDP가 backward에서 hang.

해결:
- `DistributedSampler(..., drop_last=True)`로 각 rank의 step 수를 동일하게.
- `DistributedDataParallel(..., find_unused_parameters=True)`는 정말 필요한 경우만. 가능하면 미사용 파라미터를 모델에서 제거.

## 5. MLflow가 중복 로그를 만든다

원인: 모든 rank가 `mlflow.start_run`을 호출.

해결: [`mlflow-tracking.md`](mlflow-tracking.md)의 "rank 0만 로깅" 패턴 적용.

## 6. UC Volume 저장이 너무 느리다

원인: 모든 rank가 매 save_steps마다 동시에 UC Volume에 씀.

해결: [`uc-volumes-checkpoints.md`](uc-volumes-checkpoints.md)의 "local disk + copy" 패턴. rank 0만 저장.

## 7. Serverless GPU에서 multi-node가 안 된다

원인: Serverless GPU는 single-node 환경이다. M×N 셀은 **Classic GPU 클러스터**가 필요.

해결: 매트릭스의 M×N 셀로 갈 때 Classic 클러스터 사양으로 재배포한다.

## 8. `@distributed` 안에서 `dbutils`를 못 쓴다

원인: child 프로세스에는 Databricks driver-side notebook context가 없음.

해결: `dbutils` 호출은 모두 driver(노트북) 셀에서 미리 끝낸다. 학습 함수에는 결과 값(예: 경로 문자열)만 넘긴다.

## 9. multi-node DDP인데 throughput이 안 올라간다

원인:
- 모델이 너무 작아 AllReduce 통신 비용이 compute 비중과 비슷하다.
- inter-node bandwidth(EFA 미활성)가 병목.
- batch size가 작아 GPU가 idle 상태.

해결:
- `per_device_batch_size`를 GPU 메모리가 허용하는 최대까지.
- `pin_memory=True`, `num_workers=4` 등 DataLoader 옵션 점검.
- 그래도 안 오르면 단일 노드로 충분한 워크로드일 수 있음을 인정 — 본 쿡북 M×N 셀은 **패턴 시연**이 1차 목적이다.
