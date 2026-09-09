"""VLM 기반 이상탐지 안전정지 시스템 설정값.

Jetson Thor 기준으로 로컬 vLLM 서버(OpenAI 호환 API)에 Qwen3.5-35B-A3B를
올려서 사용하는 것을 전제로 한다 (기존 Qwen2.5-VL-7B에서 교체).

중요: Jetson Thor에서 `pip install vllm`은 권장하지 않는다 (ARM64 +
Blackwell 아키텍처용 PyTorch ABI와 PyPI의 일반 vllm 휠이 맞지 않아
`ImportError: undefined symbol` 류의 에러가 난다). NVIDIA Jetson AI Lab이
Thor 전용으로 미리 빌드한 컨테이너를 공식적으로 제공하니 그걸 쓴다:

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

--network host를 쓰면 컨테이너 안의 vLLM이 호스트의 localhost:8000으로
그대로 노출되므로, main.py를 호스트에서 그냥 실행해도 아래 base_url이
그대로 동작한다 (VLM만 컨테이너, 감시 로직은 호스트).

확인된 사실 (2026-09-04, 이 로봇에서 직접 확인):
- 로컬 체크포인트 /home/dexmate/nvidia/models/Qwen3.5-35B-A3B는 멀티모달이다
  (config.json에 vision_config / image_token_id, preprocessor_config.json
  존재). 아키텍처는 Qwen3_5MoeForConditionalGeneration, BF16 67GB.
- 컨테이너의 vLLM 0.19.0 + transformers 4.57.3이 이 아키텍처를 지원한다
  (모델 config가 요구하는 transformers 4.57.0 조건 충족).
- chat_template.jinja가 enable_thinking을 처리한다
  (`enable_thinking is defined and enable_thinking is false`) -> 아래
  disable_thinking 플래그가 실제로 먹는다.

서빙 옵션에서 놓치기 쉬운 것들:
1. **--served-model-name을 주면 API가 아는 모델 id는 그 이름뿐이다.**
   로컬 경로로 서빙하면서 이 옵션이 없으면 id가 "/models/Qwen3.5-35B-A3B"가
   된다. 아래 VLMConfig.model과 어긋나면 매 호출이 404 -> ok=False ->
   fail-safe 정지가 계속 걸린다.
2. **--max-model-len을 생략하면 모델 기본값 262144이 적용된다.** 이 모델의
   KV는 토큰당 40KB(40층 x KV헤드 2 x head_dim 128 x K/V x BF16)라 최대
   길이 시퀀스 하나가 10GB다. 안전 판정은 이미지 1장 + 짧은 프롬프트라
   8192로 충분하고, 기동 시간과 메모리 프로파일링이 훨씬 가벼워진다.
3. **메모리.** Thor 통합 메모리 122GB를 CPU/ROS 스택과 공유한다. 가중치가
   67GB이므로 --gpu-memory-utilization 0.8은 98GB를 잡아 호스트+nav2
   컨테이너에 24GB만 남긴다. 0.7(85GB, KV 18GB 여유)이 안전하다.
4. **--limit-mm-per-prompt에서 video를 0으로 막아라.** 이 체크포인트는
   비디오 입력도 지원해서(video_preprocessor_config.json) vLLM이 메모리
   프로파일링 때 비디오 최악 케이스까지 잡으려 한다. 이 프로젝트는 이미지
   1장만 보낸다.
5. **--shm-size/--ulimit을 빼지 마라.** 도커 기본 /dev/shm은 64MB인데
   vLLM V1은 엔진 코어와 멀티모달 데이터를 공유 메모리로 주고받는다.
6. **판정 지연은 이미지 해상도가 아니라 출력 토큰 수가 지배한다.** 실측
   결과 1920x1200(프롬프트 2535토큰)과 640x400(495토큰)의 지연이 각각
   1.54s / 1.55s로 사실상 같았다. A3B는 활성 파라미터가 3B라 프리필이
   빠르고, 1.55초의 대부분은 출력 47토큰 디코딩(약 30 tok/s)이다.
   이 로봇의 head_camera left_rgb는 실제로 960x600(825토큰)이라 여유도
   충분하다. 따라서 지연을 줄이려면 이미지를 줄이는 게 아니라
   vlm_client.py 프롬프트의 reason 길이 제한("20단어 이내")을 줄여야 한다.
"""

from dataclasses import dataclass

# 확인된 사실 (PyPI dexcontrol 프로젝트 페이지 공식 호환성 표):
# dexcontrol과 로봇 펌웨어는 마이너 버전 단위로 락스텝(lockstep)되어 있다.
#   dexcontrol 0.X.y  <->  펌웨어 0.X.*
# 마이너 버전이 안 맞으면 서비스 콜(E-Stop 포함)이 "서비스는 보이는데 응답이
# 영원히 안 옴" 상태로 조용히 실패한다 - 타임아웃을 아무리 늘려도 해결 안 됨.
# 로봇 펌웨어가 0.5.x로 업그레이드되면서 dexcontrol 0.5.0의 E-Stop이 정상
# 동작함을 확인했다 (이전엔 펌웨어 0.4.5 + dexcontrol 0.5.0 조합에서 위
# 문제를 겪어 dexcontrol을 0.4.x로 다운그레이드했었음 - robot_interface.py
# 상단 docstring 참고). 펌웨어를 다시 업그레이드하면 이 상수와
# requirements.txt의 dexcontrol 핀을 같이 올려야 한다.
EXPECTED_DEXCONTROL_MINOR_LINE = "0.5"  # 예: "0.5"는 0.5.x 전체를 뜻함


@dataclass(frozen=True)
class EmergencyStopRetryConfig:
    """emergency_stop() 호출 자체가 실패(타임아웃/서비스 무응답)할 때의 재시도 정책.

    중요(misc.py 확인 결과): dexcontrol의 EStop.activate()는 내부 서비스
    콜이 실패해도 예외를 던지지 않고 로그만 남기고 조용히 리턴한다. 그래서
    예외를 잡는 것만으로는 실패를 감지할 수 없고, 반드시
    is_software_estop_enabled()로 실제 상태 토픽을 확인해야 한다. 또한
    이 서비스 콜의 클라이언트 타임아웃이 50ms로 매우 짧게 하드코딩되어
    있어서, 같은 프로세스에서 무거운 Python 연산(광류 계산, VLM 호출 등)이
    GIL을 붙잡고 있으면 실제 서비스는 정상인데도 타임아웃이 날 수 있다.
    """

    max_retries: int = 5
    retry_interval_s: float = 0.2  # 재시도 간격
    verify_delay_s: float = 0.1    # activate() 호출 후 상태 토픽 반영까지 대기하는 시간


@dataclass(frozen=True)
class AnomalyFilterConfig:
    """경량 1차 필터 설정. VLM을 매 프레임 호출하지 않기 위한 트리거 조건."""

    # 프레임 간 광류(optical flow) 평균 크기가 이 값을 넘으면 "움직임 급변" 후보
    flow_magnitude_threshold: float = 8.0
    # 힘/토크 센서 값의 변화율(스텝 간 차분)이 이 값을 넘으면 "충격/충돌" 후보
    force_delta_threshold_n: float = 15.0
    # 1차 필터 판단 주기 (Hz). 카메라/힘센서 스트림 자체보다 낮게 잡아 CPU 여유 확보.
    filter_rate_hz: float = 15.0
    # 카메라 스트림 시작 직후 광류가 튀는 과도구간 동안 "움직임 급변" 트리거를
    # 무시하는 시간.
    # 실측(head_camera zenoh, 로봇 완전 정지 상태): t=0.13~0.60s 구간에서
    # flow가 최대 6.35까지 올라갔고(임계값 8.0의 79%), t=1.0s 이후로는
    # 0.05~0.4로 안정됐다. 스트림/노출 안정화 전의 값이라 이 구간의 후보는
    # 허위다. 그대로 두면 기동 직후 허위 후보 -> 쿨다운 분기의 slow_down()
    # -> (속도 스케일 콜백이 없으면) 정지로 이어진다.
    # 힘 스파이크 트리거는 이 구간에도 그대로 살려둔다 (아래 anomaly_filter.py
    # 참고 - 힘 채널에는 이 과도구간이 없고, 막으면 기동 직후 충돌을 놓친다).
    startup_warmup_s: float = 1.0
    # 카메라 프레임이 이 시간보다 오래되면 "신선하지 않음"으로 보고 시각 기반
    # 판단(광류 트리거, VLM 판정)을 끈다. 힘/전류 경로는 계속 돌아간다.
    # dexcontrol의 get_obs()에는 staleness 파라미터가 없어서 스트림이 얼어도
    # 마지막 캐시 프레임을 계속 돌려준다. 확인하지 않으면 광류가 0이 되어
    # 후보가 아예 안 잡히면서 시스템은 "정상"으로 보인다 - 조용한 실명.
    # 실측(head_camera zenoh, 15Hz로 60회 읽기): age p50=6.5ms, p90=8.3ms,
    # max=38.9ms. 0.5s로 잡으면 461ms 여유가 있어 정상 스트림에서는 절대
    # 발동하지 않으면서(측정 초과 0/60), 프레임 몇 개 결손도 넘겨주고
    # 진짜 스트림 정지만 잡는다. 참고로 캡처->수신 지연은 별개로 약 78ms다
    # (timestamp_ns vs receive_time_ns 차이).
    max_frame_age_s: float = 0.5
    # 오래된 프레임 경고를 15Hz로 도배하지 않기 위한 최소 로그 간격.
    stale_warn_interval_s: float = 5.0


@dataclass(frozen=True)
class VLMConfig:
    """로컬 VLM(vLLM OpenAI 호환 서버) 연결 설정."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"  # 로컬 서버라 실제 키 불필요, 클라이언트 라이브러리 요구사항 때문에 넣음
    # vLLM을 로컬 경로(/models/Qwen3.5-35B-A3B)로 띄우면서
    # `--served-model-name qwen3.5`를 줬으므로 API가 아는 id는 "qwen3.5" 뿐이다.
    # (로컬 경로로 서빙하면 이 옵션 없이는 id가 "/models/Qwen3.5-35B-A3B"가 된다)
    # 이 값이 틀리면 매 호출이 404 -> ok=False -> fail-safe 정지로 이어진다.
    # 서버 기동 후 `curl localhost:8000/v1/models`로 반드시 대조할 것.
    model: str = "qwen3.5"
    # 이 시간 안에 응답 없으면 fail-safe(정지)로 처리.
    # 7B에서 35B-A3B로 올리면서 2.5s -> 4.0s로 늘렸다. A3B는 디코딩은 빠르지만
    # 이미지 프리필이 35B급으로 무거워서 2.5s에서는 정상 상황에도 타임아웃이
    # 나고, 그게 max_consecutive_vlm_failures를 채워 오정지로 이어진다.
    # 다만 이 호출은 감시 루프 틱 안에서 동기로 일어나므로 이 값이 곧
    # "감시 루프가 멈춰 있을 수 있는 최대 시간"이다. 무한정 늘리면 안 된다.
    request_timeout_s: float = 4.0
    max_tokens: int = 200  # JSON 판정문만 받으면 되므로 충분 (thinking은 아래에서 끈다)
    temperature: float = 0.0  # 판단 일관성을 위해 0으로 고정
    # Qwen3 계열은 하이브리드 추론 모델이라 기본적으로 <think>...</think>
    # 블록을 먼저 생성한다. 안전 루프에서는 두 가지 이유로 반드시 꺼야 한다:
    #   1) thinking 토큰이 max_tokens(200)를 다 먹어서 JSON이 아예 안 나온다.
    #   2) 나온다 해도 지연이 초 단위로 늘어나 위 타임아웃을 넘긴다.
    # vlm_client.py가 이 플래그를 chat_template_kwargs로 서버에 전달한다.
    disable_thinking: bool = True
    # VLM 호출은 워커 스레드에서 일어난다 (vlm_worker.AsyncVerifier). 워커가
    # 이 시간 안에 안 돌아오면 그 요청을 포기하고 "판단 불가"로 처리한다.
    # 정상적으로는 위 request_timeout_s에서 먼저 끊기므로 이 값은 HTTP
    # 타임아웃이 어떤 이유로든(소켓 행, 라이브러리 버그) 안 먹었을 때의
    # 마지막 안전망이다. 그래서 request_timeout_s보다 넉넉하게 잡는다.
    worker_deadline_s: float = 10.0
    # 감시 시작 전 워밍업 호출에만 쓰는 타임아웃.
    # 실측(Qwen3.5-35B-A3B / Jetson Thor): 프로세스 첫 판정 6.3s, 새 이미지
    # 해상도의 첫 판정 3.9s, 그 이후 1.5s. 즉 콜드 스타트는 위
    # request_timeout_s(4.0s)를 넘기므로 워밍업만 넉넉하게 기다려준다.
    # 이걸 짧게 잡으면 워밍업 호출 자체가 중간에 끊겨서 의미가 없어진다.
    warmup_timeout_s: float = 30.0


@dataclass(frozen=True)
class SafetyPolicyConfig:
    """VLM 판단 결과를 실제 정지/감속/계속으로 변환하는 정책."""

    stop_confidence_threshold: float = 0.75   # 이 이상이면 즉시 정지
    # 이 이상이면 감속 시도. dexcontrol의 Arm.set_joint_pos_vel()은 100Hz로
    # 계속 스트리밍돼야 하는 명령이라, "이미 실행 중인 명령"을 감속시키는
    # 개념 자체가 Arm 레벨에 없다 (src/dexcontrol/core/arm.py 확인 결과).
    # 그래서 실제 감속은 그 pos/vel 스트림을 만드는 상위 모션 루프가 속도
    # 스케일 콜백을 통해서만 할 수 있다 (robot_interface.py의
    # external_velocity_scale_setter 참고). 콜백이 없으면 이 임계값도
    # 사실상 정지를 유발한다.
    slow_confidence_threshold: float = 0.45
    # 같은 이상 상황에 대해 너무 자주 VLM을 호출하지 않도록 하는 최소 재호출 간격(초)
    vlm_cooldown_s: float = 1.0
    # VLM 호출이 이 횟수 연속으로 실패/타임아웃되면 "확인 불가" 상태로 간주하고
    # 사람이 개입하기 전까지 정지 상태를 유지
    max_consecutive_vlm_failures: int = 3


@dataclass(frozen=True)
class ForceBaselineConfig:
    """작업별 정상 힘 수준을 온라인으로 학습하는 적응형 기준선 설정.

    force_baseline.py의 AdaptiveForceBaseline이 사용한다. 고정된 절대
    임계값 대신, 이번 작업에서 지금까지 관측된 힘 분포를 기준으로 판단해서
    "가벼운 작업"과 "무거운 작업"에 같은 절대 임계값을 적용하는 문제를 피한다.
    """

    window_size: int = 200            # 기준선 계산에 쓰는 최근 샘플 개수
    min_samples: int = 30             # 기준선을 신뢰하기 전까지 필요한 최소 샘플 수
    min_mad_n: float = 0.5            # MAD 최소값 (힘이 거의 안 변하는 작업에서 나누기 폭주 방지)
    deviation_ratio_threshold: float = 5.0  # 기준선 대비 이 비율(MAD 배수) 이상 벗어나면 "이상"
    # 기준선이 아직 준비되지 않았을 때(작업 시작 직후) 쓰는 보수적 fallback 절대 임계값.
    # 이 값은 "명백히 이상한 수준"으로 넉넉하게 잡아서, 기준선이 쌓이기 전 잠깐의
    # 구간에서만 최소한의 안전망 역할을 하도록 한다.
    fallback_absolute_threshold_n: float = 25.0


@dataclass(frozen=True)
class StallDetectorConfig:
    """작업 진행 정지(stall) 후보를 판단하는 경량 1차 필터 설정.

    anomaly_filter와는 반대로 "움직임이 없음" + "힘이 낮지 않은 수준에서
    오랫동안 평평하게 유지됨"의 조합을 본다. 순간적 스파이크(충돌)가 아니라
    지속적 저항(장애물에 막힘)을 구분하기 위한 것이다.

    여기서 말하는 "힘이 낮지 않은 수준"은 절대적인 "이상 여부" 판단이 아니라
    단순히 "로봇이 뭔가에 닿아 있다"를 확인하는 최소 바닥값이다. 실제로 그
    힘 수준이 이 작업에 비정상적인지는 ForceBaselineConfig 쪽에서 판단한다.
    """

    stall_time_s: float = 4.0                    # 이 시간 동안 관찰해서 stall 여부 판단
    motion_stopped_flow_threshold: float = 2.0    # 광류 최대값이 이 이하면 "움직임 없음"
    min_force_floor_n: float = 2.0                # 힘 평균이 이 이상이면 "접촉 있음" (이상 여부와 무관)
    force_plateau_std_threshold: float = 2.0      # 힘의 표준편차가 이 이하면 "일정하게 유지"
    check_cooldown_s: float = 5.0                 # 같은 stall에 대한 VLM 재확인 최소 간격


@dataclass(frozen=True)
class TaskFeasibilityPolicyConfig:
    """작업 차단(task blocked) VLM 판단을 실제 조치로 바꾸는 정책."""

    blocked_confidence_threshold: float = 0.6
    # 작업/힘 수준과 무관하게 절대 넘으면 안 되는 물리적 안전 한계.
    # 그리퍼/팔의 정격 힘 스펙시트 기준으로 설정해야 한다. 기준선이 드리프트되는
    # 경우를 대비한 최후의 안전망이므로, 적응형 기준선과 무관하게 항상 적용된다.
    absolute_hard_force_limit_n: float = 40.0
    max_consecutive_vlm_failures: int = 3
