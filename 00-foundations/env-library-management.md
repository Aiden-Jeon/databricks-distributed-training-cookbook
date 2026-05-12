# 라이브러리 관리

분산 학습에서 가장 자주 사고가 나는 지점은 "driver에서는 import 되는데 worker child에서는 안 됨"입니다. Databricks는 라이브러리 설치 경로가 여러 개라서, 어디에 어떻게 설치하느냐가 child에서의 가시성을 결정합니다.

## 3가지 설치 경로

세 가지 설치 경로의 적용 범위와 child 가시성을 비교하면 다음과 같습니다.

| 경로 | 명령 | 적용 범위 | child 가시성 |
|------|------|----------|-------------|
| **Notebook-scoped** | `%pip install` + `%restart_python` | 노트북이 attach된 cluster의 driver와 모든 worker의 현재 Python env | TorchDistributor child도 같은 env에서 시작하므로 보임 |
| **Cluster library** | UI > Libraries 탭 또는 cluster JSON | cluster 시작 시 driver + worker에 설치 | 보임 |
| **Init script** | cluster 시작 시 실행되는 shell script | driver + worker | 보임 (`%pip`보다 빠른 단계에서 설치) |

본 쿡북은 **Notebook-scoped `%pip`**를 기본으로 사용합니다(`01-notebook-based/00-setup.ipynb`의 `%pip install --quiet "lightning==2.5.1" "nvidia-ml-py"`).

## `%pip install`이 child에 보이는 이유

직관에 반하지만 `%pip install`은 단순히 "현재 노트북에서만 보이는 venv"가 아닙니다. 실제로는 다음 단계를 거칩니다.

1. driver의 attached Python env에 패키지를 설치합니다.
2. cluster의 모든 worker에도 동기합니다(Databricks가 background에서 처리).
3. `%restart_python`으로 driver의 Python interpreter를 재시작해 새 패키지를 import할 수 있게 만듭니다.

그래서 **TorchDistributor가 띄우는 worker child 프로세스는 같은 cluster의 attached Python env에서 fresh interpreter로 시작**하므로 `%pip install`한 패키지가 자동으로 보입니다. 다음 코드가 그 동작을 확인하는 가장 짧은 예입니다.

```python
# driver
%pip install some-package
%restart_python

# worker child (TorchDistributor.run으로 띄움)
def fn():
    import some_package        # OK — 같은 env
    print(some_package.__version__)
```

## 함정 1: `%restart_python` 미실행

`%pip install` 후에 `%restart_python`을 빼먹으면 다음과 같이 어긋난 상태가 됩니다.

- 새 패키지는 디스크에 설치되어 있지만 현재 Python interpreter에는 반영되지 않습니다.
- **반면 TorchDistributor child는 fresh interpreter로 시작하므로 그쪽에서는 보입니다.**

같은 노트북에서는 import가 실패하는데 child에서는 성공하는 가장 헷갈리는 증상이 여기에서 나옵니다. `%pip install` 직후에는 항상 `%restart_python`을 함께 실행하세요.

## 함정 2: cluster restart 후 `%pip` 휘발

`%pip install`은 cluster의 라이브러리 상태에 **runtime 동안만** 남고, cluster를 restart하면 사라집니다. 영향은 다음과 같이 나타납니다.

- 매 세션마다 `00-setup.ipynb`의 `%pip` 셀을 다시 실행해야 합니다.
- Job으로 띄울 때는 setup 노트북을 먼저 실행해 동일 cluster에 패키지가 살아 있도록 보장해야 합니다.

대안으로 cluster library(UI) 또는 init script로 등록하면 cluster restart 후에도 영구적으로 유지됩니다.

## 함정 3: 의존성 충돌

DBR 17.3 LTS ML은 `torch`, `accelerate`, `transformers` 등을 **고정 버전으로 사전 설치**합니다([`env-databricks-environments.md`](env-databricks-environments.md) 표 참조). `%pip install lightning` 같은 명령은 lightning이 요구하는 `torch` 버전이 DBR 사전 설치 버전과 호환되지 않으면 torch를 downgrade·upgrade하려 시도하고, 그 과정에서 NCCL·CUDA 호환성이 깨질 수 있습니다.

권장 패턴은 다음과 같습니다.

- `%pip install --quiet "lightning==2.5.1"`처럼 **DBR과 호환되는 정확한 버전을 고정**합니다(본 쿡북 패턴).
- 큰 패키지(`torch`, `transformers`)는 DBR이 제공하는 것을 그대로 사용합니다.
- 신규 추가는 최소화합니다. 본 쿡북은 `lightning`과 `nvidia-ml-py`만 추가합니다.

## 함정 4: child가 system Python을 봄

`02-script-based/08-launch_accelerator_MxN.ipynb`의 핵심 함정입니다. `subprocess.Popen(["accelerate", ...])`로 직접 호출하면 PATH에 잡힌 `/databricks/python3/bin/accelerate`가 사용되어 **system Python env에서 실행**됩니다. 그 env는 notebook-scoped `%pip`가 설치한 패키지를 보지 못합니다.

03 행의 패키지 install이 가장 중요한 이유가 여기에 있습니다. `recommender_pkg`는 wheel로 notebook-scoped env에 설치되므로, `accelerate`가 system Python에서 도는 한 import는 실패합니다.

`03-custom-package-script-based/08-launch_accelerator_MxN.ipynb`가 채택한 해결책은 다음과 같습니다.

```python
import sys
inference_cmd = f"{sys.executable} -m accelerate.commands.accelerate_cli launch -m recommender_pkg.torch_distributor_trainer ..."
```

이 명령이 동작하는 이유를 풀어 쓰면 다음과 같습니다.

- `sys.executable`은 notebook-scoped Python의 절대 경로입니다.
- `-m accelerate.commands.accelerate_cli`로 모듈 모드 호출이 되어, notebook env에서 accelerate를 import합니다.
- `-m recommender_pkg.torch_distributor_trainer`로 entry point도 모듈 모드로 호출하면 wheel에 설치된 패키지를 정상적으로 import할 수 있고, `sys.path[0]`이 패키지 내부로 잘못 잡히는 문제도 함께 회피됩니다.

## 03 행(wheel install)이 진짜 가치 있는 이유

03-custom-package-script-based 행의 README는 "CI/CD에 적합"으로만 설명되어 있지만, 라이브러리 관리 관점에서 더 본질적인 이점이 있습니다. 02-script-based와 비교하면 다음과 같습니다.

| 이점 | 02-script-based (sys.path 보강) | 03-custom-package (wheel install) |
|------|----------------------------------|------------------------------------|
| **재현성** | `.py` 파일 내용이 곧 버전. git만 보면 충분 | wheel 파일에 버전 명시. 동일 wheel = 동일 코드 보장 |
| **worker 가시성** | `sys.path.insert(0, SCRIPT_DIR)`을 child가 직접 호출해야 함 | 일반 import. wheel이 cluster env에 있으면 어디서나 보임 |
| **버전 격리** | 노트북·환경별로 `.py`가 다를 수 있음 | `recommender_pkg==0.1.0`으로 버전 고정 |
| **타 노트북 재사용** | 폴더 복사 또는 `%run`으로만 공유 | wheel을 다른 노트북·Job에서 그대로 import |
| **subprocess 호출 (Accelerate)** | child가 system Python이면 `.py`는 보여도 sibling import 불가 | `-m recommender_pkg.xxx`로 모듈 모드 호출하면 자동 해결 |

표가 가리키는 핵심은 **"child가 다른 컨텍스트에서 도는 경우"**의 차이입니다. Accelerate launcher, Job task, MLflow recipe 같은 상황에서 `.py` 파일 기반은 매번 path 보강이 필요하고 누가 호출하느냐에 따라 동작이 달라집니다. 반면 wheel 설치는 한 번 install되면 어디서 호출되든 일관되게 동작합니다.

## 의사결정 트리

상황별로 어떤 행을 따라갈지 정리하면 다음과 같습니다.

```
PoC 단일 노트북, 코드 변경 빈번
  → 01-notebook-based (셀 안에 코드)

같은 코드를 여러 노트북에서 재사용, 같은 폴더에서만 작업
  → 02-script-based (.py + sys.path)

여러 cluster·Job에서 재사용, 또는 subprocess 기반 launcher (Accelerate) 사용,
또는 CI에서 wheel 빌드해 배포
  → 03-custom-package-script-based (wheel install)
```

## 참고

자세한 내용은 다음 자료를 참조하세요.

- [Databricks: Notebook-scoped libraries](https://docs.databricks.com/aws/en/libraries/notebooks-python-libraries)
- [Databricks: Cluster libraries](https://docs.databricks.com/aws/en/libraries/cluster-libraries)
- [Databricks: Init scripts](https://docs.databricks.com/aws/en/init-scripts/)
- [Databricks: DBR 17.3 LTS ML 사전 설치 목록](https://docs.databricks.com/aws/en/release-notes/runtime/17.3lts-ml)
- 본 쿡북 패키지 빌드: [`03-custom-package-script-based/custom_packages/README.md`](../03-custom-package-script-based/custom_packages/README.md)
