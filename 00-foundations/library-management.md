# 라이브러리 관리

분산 학습에서 가장 자주 사고가 나는 지점: "driver에서는 import 되는데 worker child에서는 안 됨." Databricks는 라이브러리 설치 경로가 여러 개라 어디에 어떻게 설치하느냐가 child에 보이는지를 결정합니다.

## 3가지 설치 경로

| 경로 | 명령 | 적용 범위 | child 가시성 |
|------|------|----------|-------------|
| **Notebook-scoped** | `%pip install` + `%restart_python` | 노트북이 attach된 cluster의 driver와 모든 worker의 현재 Python env | ✅ TorchDistributor child도 같은 env에서 시작 |
| **Cluster library** | UI > Libraries 탭 또는 cluster JSON | cluster 시작 시 driver + worker에 설치 | ✅ |
| **Init script** | cluster 시작 시 실행되는 shell script | driver + worker | ✅ (`%pip` 보다 빠른 단계에서 설치) |

본 쿡북은 **Notebook-scoped `%pip`** 를 기본으로 사용합니다 (`01-notebook-based/00-setup.ipynb` 의 `%pip install --quiet "lightning==2.5.1" "nvidia-ml-py"`).

## `%pip install` 이 child에 보이는 이유

직관에 반하지만 `%pip install` 은 단순히 "현재 노트북에서만 보이는 venv" 가 아닙니다. 실제로는:

1. driver의 attached Python env에 패키지 설치
2. cluster의 모든 worker에도 동기 (Databricks가 background에서 처리)
3. `%restart_python` 으로 driver의 Python interpreter 재시작 — 새 패키지 import 가능

→ **TorchDistributor가 띄우는 worker child 프로세스는 같은 cluster의 attached Python env에서 fresh interpreter로 시작**하므로 `%pip install` 된 패키지가 자동으로 보입니다.

검증:
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

`%pip install` 후 `%restart_python` 을 빼먹으면:
- 새 패키지는 디스크에는 설치됐지만 현재 Python interpreter에는 미반영
- **그런데 TorchDistributor child는 fresh interpreter라 보임**

→ 같은 노트북에서는 import 실패, child에서는 import 성공 — 가장 헷갈리는 증상입니다. 항상 `%restart_python` 을 직후에 실행.

## 함정 2: cluster restart 후 `%pip` 휘발

`%pip install` 은 cluster의 라이브러리 상태에 **runtime 동안만** 남습니다. cluster를 restart하면 사라집니다.

영향:
- 매 세션마다 `00-setup.ipynb` 의 `%pip` 셀을 다시 실행해야 함
- Job으로 띄울 때는 setup 노트북을 먼저 실행해 동일 cluster에 패키지가 살아 있도록 보장

대안: cluster library (UI) 또는 init script로 등록 → cluster restart 후에도 영구.

## 함정 3: 의존성 충돌

DBR 17.3 LTS ML은 `torch`, `accelerate`, `transformers` 등을 **고정 버전으로 사전 설치**합니다 (`databricks-environments.md` 표 참고). `%pip install lightning` 같은 명령은 lightning이 요구하는 `torch` 버전이 DBR 사전 설치 버전과 호환되지 않으면 torch를 downgrade/upgrade하려 시도하고, NCCL/CUDA 호환성이 깨질 수 있습니다.

권장:
- `%pip install --quiet "lightning==2.5.1"` 처럼 **DBR과 호환되는 정확한 버전을 고정** (본 쿡북 패턴)
- 큰 패키지(`torch`, `transformers`)는 DBR이 주는 것을 그대로 사용
- 신규 추가는 최소화 — 본 쿡북은 `lightning` + `nvidia-ml-py` 만 추가

## 함정 4: child가 system Python을 봄

`02-script-based/08-launch_accelerator_MxN.ipynb` 의 핵심 함정. `subprocess.Popen(["accelerate", ...])` 로 직접 호출하면 PATH의 `/databricks/python3/bin/accelerate` 가 잡혀 **system Python env에서 실행**됩니다. 그 env는 notebook-scoped `%pip` 가 설치한 패키지를 보지 못합니다.

03 행의 패키지 install이 가장 중요한 이유가 여기 있습니다. `recommender_pkg` 는 wheel로 notebook-scoped env에 설치되므로, `accelerate` 가 system Python에서 도는 한 import 실패합니다.

해결 (`03-custom-package-script-based/08-launch_accelerator_MxN.ipynb` 채택):
```python
import sys
inference_cmd = f"{sys.executable} -m accelerate.commands.accelerate_cli launch -m recommender_pkg.torch_distributor_trainer ..."
```

- `sys.executable` 은 notebook-scoped Python의 절대 경로
- `-m accelerate.commands.accelerate_cli` 로 모듈 모드 호출 → notebook env에서 accelerate import
- `-m recommender_pkg.torch_distributor_trainer` 로 entry point도 모듈 모드 → wheel에 설치된 패키지 import 가능 (`sys.path[0]` 이 패키지 내부로 잘못 잡히는 문제도 회피)

## 03 행 (wheel install)이 진짜 가치 있는 이유

03-custom-package-script-based 행의 README는 "CI/CD에 적합" 으로만 설명되어 있지만, 라이브러리 관리 관점에서 더 본질적인 이점이 있습니다:

| 이점 | 02-script-based (sys.path 보강) | 03-custom-package (wheel install) |
|------|----------------------------------|------------------------------------|
| **재현성** | `.py` 파일 내용이 곧 버전. git만 보면 충분 | wheel 파일에 버전 명시. 동일 wheel = 동일 코드 보장 |
| **worker 가시성** | `sys.path.insert(0, SCRIPT_DIR)` 을 child가 직접 호출해야 함 | 일반 import. wheel이 cluster env에 있으면 어디서나 보임 |
| **버전 격리** | 노트북·환경별로 `.py` 가 다를 수 있음 | `recommender_pkg==0.1.0` 으로 버전 고정 |
| **타 노트북 재사용** | 폴더 복사 or `%run` 으로만 공유 | wheel을 다른 노트북·Job에서 그대로 import |
| **subprocess 호출 (Accelerate)** | child가 system Python이면 `.py` 보이지만 sibling import 불가 | `-m recommender_pkg.xxx` 로 모듈 모드 호출하면 자동 해결 |

**가장 큰 차이는 "child가 다른 컨텍스트에서 도는 경우"** 입니다 — Accelerate launcher, Job task, MLflow recipe 등. `.py` 파일 기반은 매번 path 보강이 필요하고 누가 호출하느냐에 따라 동작이 다른 반면, wheel 설치는 한 번 install되면 일관됩니다.

## 의사결정 트리

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

- Databricks Notebook-scoped libraries: https://docs.databricks.com/aws/en/libraries/notebooks-python-libraries
- Cluster libraries: https://docs.databricks.com/aws/en/libraries/cluster-libraries
- Init scripts: https://docs.databricks.com/aws/en/init-scripts/
- DBR 17.3 LTS ML 사전 설치 목록: https://docs.databricks.com/aws/en/release-notes/runtime/17.3lts-ml
- 본 쿡북 패키지 빌드: [`03-custom-package-script-based/custom_packages/README.md`](../03-custom-package-script-based/custom_packages/README.md)
