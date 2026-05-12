# 인증 (Authentication)

분산 학습에서 child 프로세스가 Databricks API(MLflow, UC, SQL warehouse 등)에 접근하려면 자격증명이 필요합니다. 본 쿡북은 **PAT(Personal Access Token) 전달 패턴**을 기본으로 합니다. driver에서 토큰을 한 번 읽고, TorchDistributor child에 인자로 넘겨 환경변수로 설정하는 방식입니다.

이 문서는 PAT 패턴의 가정과 한계, 그리고 실제 운영(특히 service principal로 도는 Job)에서 무엇을 바꿔야 하는지를 정리합니다.

## 본 쿡북의 기본 패턴 — PAT 전달

`01-notebook-based/00-setup.ipynb`에서 driver의 자격증명을 다음과 같이 추출합니다.

```python
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
DB_HOST = ctx.extraContext().apply("api_url")
DB_TOKEN = ctx.apiToken().get()
```

학습 함수에 인자로 넘깁니다.

```python
TorchDistributor(...).run(train_fn_ddp, db_host=DB_HOST, db_token=DB_TOKEN, ...)
```

child 안에서는 환경변수에 다시 설정합니다.

```python
os.environ["DATABRICKS_HOST"] = db_host
os.environ["DATABRICKS_TOKEN"] = db_token
```

이렇게 두면 MLflow, `WorkspaceClient`, `databricks-sdk` 등이 환경변수에서 자격증명을 자동으로 찾습니다.

## 왜 child가 driver의 자격증명을 자동 상속하지 않는가

[`debug-common-pitfalls.md §2-1`](debug-common-pitfalls.md)의 설명을 보강합니다. TorchDistributor child는 모드별로 다음과 같이 동작합니다.

- `local_mode=True`는 driver의 subprocess이지만 **fresh Python interpreter**입니다. driver Python 프로세스의 메모리(`dbutils` 객체, MLflow tracking client)는 상속되지 않고 환경변수만 OS 레벨에서 상속됩니다.
- `local_mode=False`는 다른 머신(worker 노드)의 executor task로 도는 형태입니다. driver와 완전히 다른 OS 프로세스라 더더욱 상속이 없습니다.

여기에 `dbutils.notebook.entry_point`는 driver의 py4j gateway를 통해서만 접근 가능하므로 child 프로세스에서는 호출 자체가 불가능합니다([`debug-common-pitfalls.md §8`](debug-common-pitfalls.md)).

따라서 **driver 셀에서 `dbutils`로 자격증명을 추출한 뒤 인자로 child에 명시 전달**하는 것이 유일한 길입니다.

## ctx.apiToken()이 반환하는 토큰의 정체

같은 코드라도 실행 컨텍스트에 따라 반환되는 토큰이 다릅니다. 다음 표가 그 차이를 정리합니다.

| 실행 컨텍스트 | `ctx.apiToken()`이 반환하는 것 |
|--------------|-------------------------------|
| Interactive 노트북 (사용자가 attach) | 사용자의 임시 PAT (수명: 노트북 세션) |
| Job — `run_as`가 사용자 | 그 사용자의 임시 PAT |
| Job — `run_as`가 service principal | service principal의 임시 OAuth M2M 토큰 (단, 노트북 컨텍스트에서는 PAT처럼 노출) |

`ctx.apiToken()` 자체는 PAT와 service principal 모두에서 동작하므로 **본 쿡북 패턴은 두 경우 모두에서 그대로 작동**합니다. 다만 토큰의 종류와 수명이 다르다는 점은 알아 두어야 합니다.

## 한계 1: 토큰 만료

`ctx.apiToken()`으로 얻은 토큰은 **단기 토큰**입니다. 수명은 컨텍스트별로 다르지만 보통 수십 분에서 수 시간 정도입니다.

- 학습이 토큰 수명보다 길어지면 child의 MLflow 호출이 `401 Unauthorized`로 떨어집니다.
- 본 쿡북의 권장 budget이 15분이라 이 문제는 드물지만, 실제 워크로드에서 1시간 이상 학습은 위험합니다.

회피책은 세 가지를 고려할 수 있습니다.

1. **장기 PAT 발급.** 사용자 settings에서 90일짜리 PAT를 생성한 뒤 secret으로 등록하고, `dbutils.secrets.get("scope", "key")`로 로드합니다. 본 쿡북 패턴의 `ctx.apiToken()`만 교체하면 됩니다.
2. **MLflow는 토큰 만료 시 재인증이 어렵습니다.** 학습 중간에 child가 한 번 더 자격증명을 refresh할 수단이 없으므로 1번 방식이 단순합니다.
3. **OAuth M2M(service principal).** 토큰을 SDK가 자동으로 갱신해 줍니다. 아래의 "OAuth U2M / M2M" 섹션을 참고하세요.

## 한계 2: PAT 보안

토큰을 함수 인자로 넘기면 다음과 같은 노출 경로가 생깁니다.

- driver → child 직렬화(cloudpickle) 과정에 토큰 문자열이 들어갑니다.
- worker 노드 메모리에 plaintext로 존재합니다.
- MLflow `log_param` 등에 실수로 들어가면 로그에 평문이 그대로 노출됩니다.

회피책은 다음과 같습니다.

- `db_token`을 **MLflow params/tags에 절대 로그하지 않습니다**(현재 trainer는 안전한 것으로 검증되어 있습니다).
- worker stdout/stderr에 토큰이 찍히지 않는지 확인합니다.
- 가장 깔끔한 방법은 service principal + OAuth M2M으로 전환해 PAT 자체를 쓰지 않는 것입니다.

## Service Principal로 도는 Job

Workflow를 `run_as = <service_principal>`로 설정하면 다음과 같이 동작합니다.

1. `ctx.apiToken()`이 해당 SP의 임시 토큰을 반환하므로 본 쿡북 코드는 변경 없이 동작합니다.
2. **SP가 필요한 권한**을 미리 grant해 두어야 합니다.
   - UC catalog/schema/volume: `USE CATALOG`, `USE SCHEMA`, `READ VOLUME`, `WRITE VOLUME`
   - MLflow experiment: experiment 경로에 `CAN EDIT`
   - UC Model Registry(등록 시): schema에 `CREATE MODEL`
   - Cluster: SP가 cluster를 attach할 수 있어야 합니다(cluster policy 또는 permissions).
3. SP는 personal workspace 폴더가 없으므로 experiment 경로를 `/Users/<sp-uuid>/...`가 아닌 **공유 폴더**(`/Shared/...` 또는 workspace-level)로 두어야 합니다.

본 쿡북의 `EXPERIMENT_PATH = f"/Users/{USERNAME}/recommender-notebook-based"`는 사용자 기준입니다. SP로 돌릴 때는 `EXPERIMENT_PATH = "/Shared/recommender-distributed-training"` 같은 경로로 교체합니다.

## OAuth U2M / M2M

장기적으로는 PAT 대신 OAuth가 권장됩니다. 사용처를 비교하면 다음과 같습니다.

| 방식 | 사용처 | 본 쿡북에 적용 |
|------|--------|--------------|
| **U2M (User-to-Machine)** | 사용자가 로컬에서 `databricks auth login`으로 인증 | 로컬 개발용. 본 쿡북은 노트북 안에서 실행되므로 불필요 |
| **M2M (Machine-to-Machine)** | service principal + OAuth client_id/secret | Job, CI/CD. SDK가 토큰을 자동 갱신 |

M2M으로 전환할 때의 driver 코드 예시는 다음과 같습니다.
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

child 함수에 전달할 때는 다음과 같이 환경변수에 다시 설정합니다.

```python
def train_fn(..., client_id, client_secret):
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_CLIENT_ID"] = client_id
    os.environ["DATABRICKS_CLIENT_SECRET"] = client_secret
    # databricks-sdk가 자동으로 OAuth flow 수행, 토큰 만료 시 갱신
```

장점은 토큰 만료를 신경 쓰지 않아도 되고 secret rotation을 secret scope 업데이트만으로 끝낼 수 있다는 점입니다. 단점은 패턴이 더 복잡해지고 본 쿡북의 PAT 패턴과 호환되지 않는다는 점입니다.

## 권한 함정 — UC Model Registry

학습 종료 후 `mlflow.register_model(...)`로 UC Model Registry에 등록할 때 자주 막히는 권한을 정리합니다.

| 권한 | 부여 대상 |
|------|----------|
| `USE CATALOG` | catalog (예: `main`) |
| `USE SCHEMA` | schema (예: `main.distributed_cookbook`) |
| `CREATE MODEL` | schema |
| `EXECUTE` | 등록된 model을 다른 사용자가 load할 때 |

PAT나 사용자 계정으로 실행할 때는 그 사용자에게, SP로 실행할 때는 SP에 위 권한이 grant되어 있어야 합니다.

자주 보는 에러는 다음과 같습니다.

- `PERMISSION_DENIED: User does not have CREATE MODEL on schema 'main.distributed_cookbook'`
- 해결은 `` GRANT CREATE MODEL ON SCHEMA main.distributed_cookbook TO `<user-or-sp>` `` 실행입니다.

## 의사결정 트리

상황별로 어떤 방식을 고를지 정리하면 다음과 같습니다.

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

자세한 내용은 다음 자료를 참조하세요.

- [Databricks: Authentication overview](https://docs.databricks.com/aws/en/dev-tools/auth)
- [Databricks: Service principal OAuth (M2M)](https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m)
- [Databricks: `dbutils.secrets` workflow](https://docs.databricks.com/aws/en/security/secrets/example-secret-workflow)
- [Databricks: UC Model Registry 권한](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/index)
- 본 쿡북의 자격증명 함정: [`debug-common-pitfalls.md` §2-1](debug-common-pitfalls.md)
