# 인증 (Authentication)

분산 학습에서 child 프로세스가 Databricks API (MLflow, UC, SQL warehouse 등)에 접근하려면 자격증명이 필요합니다. 본 쿡북은 **PAT (Personal Access Token) 전달 패턴**을 기본으로 합니다 — driver에서 토큰을 한 번 읽고, TorchDistributor child에 인자로 넘겨 환경변수로 설정.

이 문서는 PAT 패턴의 가정, 한계, 그리고 실제 운영(특히 service principal로 도는 Job)에서 무엇을 바꿔야 하는지 정리합니다.

## 본 쿡북의 기본 패턴 — PAT 전달

`01-notebook-based/00-setup.ipynb`:
```python
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
DB_HOST = ctx.extraContext().apply("api_url")
DB_TOKEN = ctx.apiToken().get()
```

학습 함수에 인자로 넘겨:
```python
TorchDistributor(...).run(train_fn_ddp, db_host=DB_HOST, db_token=DB_TOKEN, ...)
```

child 안에서:
```python
os.environ["DATABRICKS_HOST"] = db_host
os.environ["DATABRICKS_TOKEN"] = db_token
```

→ MLflow, `WorkspaceClient`, `databricks-sdk` 등이 환경변수에서 자격증명을 자동으로 찾습니다.

## 왜 child가 driver의 자격증명을 자동 상속하지 않는가

`debug-common-pitfalls.md §2-1` 의 설명을 보강합니다. TorchDistributor child는:

- `local_mode=True` → driver의 subprocess이지만 **fresh Python interpreter**. driver Python 프로세스의 메모리(`dbutils` 객체, MLflow tracking client)는 상속되지 않습니다. 환경변수만 OS 레벨에서 상속됩니다.
- `local_mode=False` → 다른 머신(worker 노드)의 executor task. driver와 완전히 다른 OS 프로세스라 더더욱 상속 없음.

그리고 `dbutils.notebook.entry_point` 는 driver의 py4j gateway를 통해서만 접근 가능 — child 프로세스에서는 호출 자체가 안 됩니다 (`debug-common-pitfalls.md §8`).

따라서 **driver 셀에서 `dbutils` 로 자격증명을 추출** → **인자로 child에 명시 전달** 이 유일한 길입니다.

## ctx.apiToken() 이 반환하는 토큰의 정체

| 실행 컨텍스트 | `ctx.apiToken()` 이 반환하는 것 |
|--------------|-------------------------------|
| Interactive 노트북 (사용자가 attach) | 사용자의 임시 PAT (수명: 노트북 세션) |
| Job — `run_as` 가 사용자 | 그 사용자의 임시 PAT |
| Job — `run_as` 가 service principal | service principal의 임시 OAuth M2M 토큰 (단, 노트북 컨텍스트에서는 PAT처럼 노출) |

→ `ctx.apiToken()` 자체는 PAT 와 service principal 둘 다에서 동작합니다. **본 쿡북 패턴은 두 경우 모두에서 그대로 동작**합니다. 다만 토큰의 종류와 수명이 다릅니다.

## 한계 1: 토큰 만료

`ctx.apiToken()` 으로 얻은 토큰은 **단기 토큰** (수명은 컨텍스트 별로 다르지만 보통 수십 분 ~ 수 시간):

- 학습이 토큰 수명보다 길어지면 child의 MLflow 호출이 `401 Unauthorized` 로 죽습니다.
- 본 쿡북 권장 budget이 15분이라 이 문제가 드물지만, 실 워크로드에서 1시간+ 학습은 위험.

회피책:
1. **장기 PAT 발급**: 사용자 settings에서 90일짜리 PAT 생성 후 secret으로 등록 → `dbutils.secrets.get("scope", "key")` 로 로드. 본 쿡북 패턴의 `ctx.apiToken()` 만 교체.
2. **MLflow는 토큰 만료 시 재인증**: 학습 중간에 child가 한 번 더 자격증명을 refresh할 수단이 없으므로 1번이 단순합니다.
3. **OAuth M2M (service principal)**: 토큰을 SDK가 자동 갱신 — 아래 §"OAuth M2M" 참고.

## 한계 2: PAT 보안

토큰을 함수 인자로 넘기면:
- driver → child 직렬화 (cloudpickle) 과정에 토큰 문자열이 들어감
- worker 노드 메모리에 plaintext로 존재
- MLflow `log_param` 등에 실수로 들어가면 로그에 평문 노출

회피책:
- `db_token` 을 **MLflow params/tags에 절대 로그하지 않기** (현재 trainer는 안전, 검증 완료)
- worker stdout/stderr에 토큰이 찍히지 않는지 확인
- 가장 좋은 것은 service principal + OAuth M2M으로 PAT 자체를 안 쓰는 것

## Service Principal로 도는 Job

Workflow를 `run_as = <service_principal>` 로 설정하면:

1. `ctx.apiToken()` 은 그 SP의 임시 토큰을 반환 → 본 쿡북 코드 변경 없이 동작
2. **SP가 필요한 권한**을 미리 grant해야 함:
   - UC catalog/schema/volume: `USE CATALOG`, `USE SCHEMA`, `READ VOLUME`, `WRITE VOLUME`
   - MLflow experiment: `CAN EDIT` on experiment path
   - UC Model Registry (등록 시): `CREATE MODEL` on schema
   - Cluster: SP가 cluster를 attach할 수 있어야 함 (cluster policy / permissions)
3. SP는 personal workspace 폴더가 없으므로 experiment 경로를 `/Users/<sp-uuid>/...` 가 아닌 **공유 폴더**(`/Shared/...` 또는 workspace-level)로 둬야 함

본 쿡북의 `EXPERIMENT_PATH = f"/Users/{USERNAME}/recommender-notebook-based"` 는 사용자 기준입니다. SP로 돌릴 때는 `EXPERIMENT_PATH = "/Shared/recommender-distributed-training"` 같은 경로로 교체.

## OAuth U2M / M2M

장기적으로는 PAT 대신 OAuth가 권장됩니다.

| 방식 | 사용처 | 본 쿡북에 적용 |
|------|--------|--------------|
| **U2M (User-to-Machine)** | 사용자가 로컬에서 `databricks auth login` 으로 인증 | 로컬 개발 시. 본 쿡북은 노트북 안에서 실행되므로 불필요 |
| **M2M (Machine-to-Machine)** | service principal + OAuth client_id/secret | Job, CI/CD. SDK가 토큰 자동 갱신 |

M2M으로 전환 시:
```python
from databricks.sdk import WorkspaceClient

# client_id, client_secret을 secret으로 로드
w = WorkspaceClient(
    host=DB_HOST,
    client_id=dbutils.secrets.get("scope", "sp_client_id"),
    client_secret=dbutils.secrets.get("scope", "sp_client_secret"),
)
# w.config 에서 토큰을 받아 환경변수 세팅
```

child 함수에 전달할 때:
```python
def train_fn(..., client_id, client_secret):
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_CLIENT_ID"] = client_id
    os.environ["DATABRICKS_CLIENT_SECRET"] = client_secret
    # databricks-sdk가 자동으로 OAuth flow 수행, 토큰 만료 시 갱신
```

장점: 토큰 만료 걱정 없음, secret rotation은 secret scope 업데이트만.
단점: 패턴이 복잡해지고 본 쿡북 PAT 패턴과 비호환.

## 권한 함정 — UC Model Registry

학습 종료 후 `mlflow.register_model(...)` 로 UC Model Registry에 등록할 때 자주 막히는 권한:

| 권한 | 부여 대상 |
|------|----------|
| `USE CATALOG` | catalog (예: `main`) |
| `USE SCHEMA` | schema (예: `main.distributed_cookbook`) |
| `CREATE MODEL` | schema |
| `EXECUTE` | 등록된 model을 다른 사용자가 load 시 |

PAT/사용자로 실행 시 사용자에게 위 권한이 있어야 합니다. SP로 실행 시 SP에 grant 필요.

흔한 에러:
- `PERMISSION_DENIED: User does not have CREATE MODEL on schema 'main.distributed_cookbook'`
- → `GRANT CREATE MODEL ON SCHEMA main.distributed_cookbook TO \`<user-or-sp>\`` 실행

## 의사결정 트리

```
Interactive 노트북, < 1시간 학습
  → ctx.apiToken() 그대로 (본 쿡북 기본)

Job, run_as = 사용자
  → 동일. 단, 토큰 만료 위험이 있으면 PAT secret으로 교체

Job, run_as = service principal
  → 동일 코드, EXPERIMENT_PATH를 /Shared/... 로 교체, SP에 권한 grant

장기 운영 / CI / 토큰 자동 갱신 필요
  → OAuth M2M (client_id/secret을 secret scope에 보관)
```

## 참고

- Databricks authentication overview: https://docs.databricks.com/aws/en/dev-tools/auth
- Service principal OAuth: https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m
- `dbutils.secrets`: https://docs.databricks.com/aws/en/security/secrets/example-secret-workflow
- UC Model Registry 권한: https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/index
- 본 쿡북의 자격증명 함정: [`debug-common-pitfalls.md` §2-1](debug-common-pitfalls.md)
