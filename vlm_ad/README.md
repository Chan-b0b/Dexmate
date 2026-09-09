# VLM 기반 로봇 안전정지 시스템 (Dexmate Vega / Jetson Thor)

## 구성 파일

| 파일 | 역할 |
|---|---|
| `config.py` | 모든 임계값/설정 |
| `anomaly_filter.py` | 경로 1 - 경량 1차 이상 후보 필터 (광류+힘 스파이크) |
| `vlm_client.py` | VLM 호출 클라이언트 (이상탐지 + 작업차단판단 두 종류) |
| `vlm_worker.py` | VLM 호출을 워커 스레드로 분리 (감시 루프가 응답을 기다리며 멈추지 않게) |
| `calibrate_filter.py` | 1차 필터 임계값을 실제 카메라 프레임으로 튜닝하는 도구 (센서 읽기 전용) |
| `tests/` | 회귀 테스트 (`python tests/run_all.py`) - 아래 "테스트" 절 참고 |
| `stall_detector.py` | 경로 2 - 작업 진행 정지(stall) 패턴 감지 |
| `force_baseline.py` | 작업별 적응형 힘/전류 기준선 (스칼라+벡터) |
| `arm_current_monitor.py` | 경로 3 - 관절별 전류 이상탐지 |
| `momentum_observer.py` | (선택, 미완성) URDF 기반 운동량 관측기 - 모터 토크 상수 확인 전까지는 참고용 |
| `vega_dynamics.py` | (선택) dexmate_urdf에서 Vega 동역학 모델 로드 |
| `robot_interface.py` | dexcontrol 연동 어댑터 (확인된 API 반영) |
| `safety_supervisor.py` | 3개 경로를 조율하는 메인 감시 루프 |
| `main.py` | 실행 진입점 |
| `resume_robot.py` | Software E-Stop 해제 전용 도구 (사람의 명시적 확인 필요) |
| `requirements.txt` | 의존성 |

## 실행 전 준비

1. **로컬 VLM 서버 실행** (Jetson Thor, Docker 필수 — `pip install vllm`은 ARM64+Blackwell
   PyTorch ABI 불일치로 `ImportError: undefined symbol` 에러가 나서 권장하지 않음)
   ```bash
   sudo docker run -it --rm --runtime=nvidia --network host \
       --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
       -v /home/dexmate/nvidia/models:/models \
       ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
       vllm serve /models/Qwen3.5-35B-A3B \
           --served-model-name qwen3.5 \
           --max-model-len 8192 \
           --gpu-memory-utilization 0.7 \
           --limit-mm-per-prompt '{"image":1,"video":0}' \
           --enable-prefix-caching
   ```
   `--network host`를 쓰므로 **VLM은 컨테이너, `main.py`는 호스트**에서 돌려도
   `config.py`의 `localhost:8000` 설정이 그대로 동작한다.

   **놓치기 쉬운 것** (자세한 내용은 `config.py` 상단 주석):
   - `--served-model-name qwen3.5`를 줬으면 `VLMConfig.model`도 `"qwen3.5"`여야
     한다. 어긋나면 매 호출이 404 → `ok=False` → fail-safe 정지가 계속 걸린다.
     `curl localhost:8000/v1/models`로 대조할 것.
   - `--max-model-len`을 생략하면 모델 기본값 262144가 쓰인다 (KV 토큰당 40KB →
     최대 길이 시퀀스 하나가 10GB). 8192로 충분하다.
   - BF16 가중치가 67GB인데 Thor 통합 메모리 122GB를 ROS 스택과 공유한다.
     `--gpu-memory-utilization 0.8`은 98GB를 잡아 호스트+nav2에 24GB만 남긴다 →
     0.7 권장.
   - `video`를 0으로 막지 않으면 vLLM이 메모리 프로파일링 때 비디오 최악
     케이스까지 잡는다 (이 체크포인트는 비디오 입력도 지원한다).
   - **thinking 모드는 코드에서 끈다** (`VLMConfig.disable_thinking=True` →
     `chat_template_kwargs`). 켜두면 thinking 토큰이 `max_tokens`를 다 먹어
     JSON이 안 나오고 지연도 타임아웃을 넘긴다. 이 체크포인트의
     `chat_template.jinja`가 이 옵션을 지원하는 것은 확인했다.

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```
   `momentum_observer.py`/`vega_dynamics.py`를 실제로 쓰려면 추가로:
   ```bash
   pip install pin dexmate-urdf
   ```

3. **`main.py` 확인/조정**
   - `build_robot_adapter()`가 `dexcontrol.robot.Robot()`을 그대로 씀 (확인됨)
   - `DexcontrolAdapter`의 `monitor_arms`, `external_velocity_scale_setter` 등 필요에 맞게 조정

4. **실행**
   ```bash
   python main.py
   ```
   기동 시 `supervisor.warm_up()`이 실제 카메라 프레임으로 VLM을 한 번씩
   미리 호출한다 (두 경로 합산 약 4초). 콜드 스타트 판정이
   `request_timeout_s`를 넘겨(실측 6.3초) **기동 후 첫 이상 상황의 판정이
   날아가는 것**을 막기 위한 것이고, 동시에 모델 id 오타나 서버 미기동을
   기동 시점에 잡아내는 자기진단 역할도 한다 (실패 시 CRITICAL 로그, 단
   기동 자체를 막지는 않는다 - VLM이 죽어도 전류/힘 경로는 살아 있다).

## 1차 필터 임계값 튜닝 (실제 카메라 프레임)

`config.py`의 `flow_magnitude_threshold`/`force_delta_threshold_n`은 예시값이다.
Farneback 광류의 값 범위는 장면의 질감·조명·카메라 흔들림에 따라 완전히
달라져서 합성 이미지로는 튜닝이 불가능하다. `calibrate_filter.py`는 센서만
읽고 로봇에 명령은 보내지 않는다.

```bash
# 1) 정상 기준선 - 반드시 로봇이 "평소 하는 작업"을 하는 동안 돌린다.
#    정지 상태로 수집하면 flow가 0.05 수준이라 임계값이 비현실적으로 낮게
#    나오고 실전에서 오탐이 폭주한다.
python calibrate_filter.py record --label normal --duration 300 \
    --task "배터리 케이스를 빈에서 오른팔로 집는 중"

# 2) 이상 상황 연출 (안전하게 재현 가능한 것만)
python calibrate_filter.py record --label person_approach --duration 60

# 3) 임계값 산출 - 정상 후보율(/분) vs 이상 검출율(%) 표를 보고 고른다
python calibrate_filter.py report

# 4) 저장된 실제 프레임을 VLM에 넣어 2차 판정 오탐 확인
python calibrate_filter.py replay --label normal --only-candidates
```

`report`는 정상 후보율이 `--max-false-rate`(기본 1회/분) 이하가 되는 가장
낮은 임계값을 권장값으로 제시한다. 임계값을 올리면 오탐이 줄고 미탐이 늘기
때문에, 최종 선택은 "이 작업에서 무엇을 놓치면 안 되는지"로 판단해야 한다.

## E-Stop 해제 (재개)

**의도적으로 `main.py`/`safety_supervisor.py`에는 자동 해제 로직이 없다.**
정지는 소프트웨어가 판단해서 걸 수 있지만, 재개는 반드시 사람이 상황을
확인한 뒤 명시적으로 실행해야 한다는 원칙 때문이다.

```bash
python resume_robot.py
```

정확한 확인 문구(`RESUME ROBOT`)를 입력해야 진행되고, 해제 후에는
`is_software_estop_enabled()`로 실제로 반영됐는지 다시 확인한다 (서비스
콜 자체가 조용히 실패할 수 있다는 걸 이미 겪었으므로).

## 테스트

```bash
python tests/run_all.py        # 전체 (요약)
python tests/run_all.py -v     # 전체 (각 테스트 출력까지)
python tests/test_warmup.py    # 개별 실행도 가능
```

pytest를 쓰지 않는다 - 이 프로젝트에 테스트 프레임워크 의존성이 없고, 실제
서버/로봇을 상대로 손으로 돌려보는 스크립트 성격이라 그냥 `python`으로 실행한다.

| 테스트 | 검증 내용 | 실제 서버 필요 |
|---|---|---|
| `test_vlm_worker.py` | 워커의 비블로킹 poll, 백프레셔, deadline 포기, 뒤늦은 응답 폐기, 예외 흡수 | 아니오 |
| `test_supervisor_async.py` | 비동기 배선 후에도 정책이 유지되는지 (감속/정지/연속실패/작업차단/절대힘한계) | 아니오 |
| `test_warmup.py` | 콜드 스타트 워밍업, 워밍업 전용 타임아웃, 모델 id 자기진단 | **예** (없으면 SKIP) |
| `test_frame_staleness.py` | 오래된 프레임에서 시각 경로만 끄고 힘/전류 경로가 살아있는지 | 아니오 |

**어느 테스트도 실제 로봇을 필요로 하지 않는다** (전부 가짜 로봇 어댑터).
`tests/fixtures/head_camera_frame.jpg`는 실제 head_camera left_rgb에서 받은
960x600 프레임이다 - VLM 콜드 스타트 비용이 이미지 해상도별로 따로 발생하므로
합성 이미지가 아니라 실제 해상도의 프레임으로 테스트해야 한다.

> **vLLM 서버 콘솔에 404가 찍히는 이유**
> `test_warmup.py`의 3번 항목이 "모델 id 오타를 기동 시점에 잡아내는가"를
> 검증하려고 **의도적으로 존재하지 않는 모델 이름**으로 요청을 보낸다.
> `warm_up()`이 두 경로를 부르므로 실행 1회당 404가 2건 찍힌다 — 정상이며
> 무해하다. 로그만 보고 알아볼 수 있게 모델 이름을
> `intentionally-missing-model-warmup-selftest`로 지어 놨다.
> ```
> ERROR ... error=ErrorInfo(message='The model `intentionally-missing-model-warmup-selftest`
>                          does not exist.', type='NotFoundError', param='model', code=404)
> INFO: ... "POST /v1/chat/completions HTTP/1.1" 404 Not Found
> ```

## 알려진 미해결/확인 필요 사항 (정직하게 남겨둔 것)

- **[해결됨] dexcontrol ↔ 로봇 펌웨어 버전 락스텝**: dexcontrol과 펌웨어는
  마이너 버전 단위로 락스텝되어 있다 (`dexcontrol 0.X.y` ↔ `펌웨어 0.X.*`,
  PyPI 공식 호환성 표 기준). 한때 펌웨어 0.4.5 + dexcontrol 0.5.0 조합에서
  E-Stop이 응답 없이 멈춰 dexcontrol을 0.4.x로 다운그레이드했었으나, 이후
  로봇 펌웨어가 0.5.x로 업그레이드되어 dexcontrol 0.5.0의 E-Stop이 정상
  동작함을 확인했다. 지금은 `dexcontrol>=0.5.0,<0.6.0`에 고정돼 있다.
  **펌웨어를 다시 업그레이드하면 `requirements.txt`와 `config.py`의
  `EXPECTED_DEXCONTROL_MINOR_LINE`를 같이 올려야 한다** - 안 그러면 E-Stop
  같은 서비스 콜이 "서비스는 보이는데 응답이 영원히 안 옴" 상태로 조용히
  실패한다 (타임아웃을 늘려도 소용없음). `main.py`가 시작 시 버전을 확인해서
  어긋나면 CRITICAL 로그를 남긴다.
- 감속(slow_down)은 검증된 부분 감속 API가 없어 기본적으로 정지로 대체됨
  (`robot_interface.py` 상단 주석, `external_velocity_scale_setter`로 실제
  모션 루프와 연동 가능)
- `momentum_observer.py`는 전류→토크 변환에 필요한 모터 토크 상수가
  확인되지 않아 실전에는 `arm_current_monitor.py`(전류 기반 적응형 기준선)를
  우선 사용. 토크 상수를 구하면 momentum observer로 업그레이드 가능.
- 모든 힘/전류 관련 임계값(`config.py`)은 예시값이므로 실제 로봇/작업에
  맞게 튜닝 필요.
- `anomaly_filter.py`의 `flow_magnitude_threshold`(8.0)는 아직 실제 작업
  중의 프레임으로 튜닝되지 않았다. `calibrate_filter.py`로 수집/산출한다
  (아래 "1차 필터 임계값 튜닝" 참고). 힘 스파이크 경로(15N)는 확인됨.
- **[해결됨] 카메라 스트림 시작 직후 약 0.6초간 광류가 튄다** (실측: 로봇이
  완전히 정지한 상태에서 flow가 최대 6.35까지, 임계값 8.0의 79%). t=1.0s
  이후로는 0.05~0.4로 안정된다. 임계값을 낮게 잡으면 기동 직후 허위 후보 →
  쿨다운 분기의 `slow_down()` → (속도 콜백이 없으면) 정지로 이어진다.
  `AnomalyFilterConfig.startup_warmup_s`(1.0초) 동안 **광류 트리거만**
  무시하도록 했다. 힘 스파이크 트리거는 이 구간에도 살려둔다 - 힘 채널에는
  과도구간이 없고(정지 상태 max 0.54N), 같이 막으면 기동 직후 충돌을
  놓치는 구멍이 생긴다.
- **[해결됨] 카메라 프레임 신선도(staleness)**: dexcontrol의 `get_obs()`는
  스트림이 얼어도 마지막 캐시 프레임을 계속 돌려준다(staleness 파라미터가
  없음). 확인하지 않으면 광류가 0이 되어 후보가 아예 안 잡히면서 시스템은
  "정상"으로 보인다 - 조용한 실명. 이제 `include_timestamp=True`로 받아
  `receive_time_ns`(로컬 `time.time_ns()`와 같은 시계) 기준 age를 확인한다.
  **정책: 오래된 프레임에서 정지시키지 않는다** - 시각 기반 신호만 무효로
  하고(광류 트리거, stall 후보, VLM 판정) 힘/전류 경로로 계속 감시하며
  경고를 반복 기록한다. 카메라가 잠깐 끊기는 것 자체가 위험을 뜻하지 않고,
  그때마다 정지시키면 운영이 불가능하기 때문이다. 힘의 절대 안전 한계
  검사만은 후보 여부와 무관하게 항상 돌아간다.
- VLM 판정 지연은 실측 약 1.55초이고, 그 대부분이 **출력 토큰 디코딩**
  (47토큰, 약 30 tok/s)이다. 해상도는 지연을 거의 바꾸지 않는다:
  1920x1200(2535토큰) 1.54s vs 640x400(495토큰) 1.55s. 실제 카메라
  프레임은 960x600(825토큰)이다. 지연을 줄이려면 프롬프트의 `reason` 길이
  제한(현재 "20단어 이내")을 줄이는 쪽이 효과적이다.

## 아키텍처 한눈에 보기

```
카메라 프레임 + 힘/토크 + 관절 전류
        │
        ├─ 경로 1: LightweightAnomalyFilter → [워커] VLMAnomalyVerifier → stop/slow
        ├─ 경로 2: StallDetector + AdaptiveForceBaseline → [워커] TaskFeasibilityVerifier → pause_task
        └─ 경로 3: ArmCurrentAnomalyDetector (관절별 적응형 기준선) → stop
```

`[워커]` 표시된 VLM 호출은 **감시 루프 밖(워커 스레드)에서** 일어난다. 감시 루프가
VLM 응답(최대 `request_timeout_s`)을 기다리며 멈추면 그 동안 관절 전류 이상탐지,
힘 스파이크 감지, 정지 미확인 상태 복구 재시도가 모두 멈추고, 게다가 dexcontrol
E-Stop 서비스 콜의 클라이언트 타임아웃이 50ms라 **정지 호출 자체가 실패할 수
있다**. 그래서 두 경로는 `poll()`(판정 수거) → `submit()`(요청 제출) 구조이고,
판정은 요청한 틱이 아니라 몇 틱 뒤에 반영된다. 자세한 정책(하나만 in-flight,
deadline 초과 시 요청 포기, 뒤늦은 응답 폐기)은 `vlm_worker.py` docstring 참고.

절대 힘 한계 초과(`absolute_hard_force_limit_n`)만은 예외로 VLM을 거치지 않고
그 틱에서 즉시 정지한다.
