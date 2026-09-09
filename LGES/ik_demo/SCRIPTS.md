# ik_demo 실행 스크립트

각 모듈의 실행 방법과 용도를 정리한 간단한 가이드입니다.

## 빠른 참조

| 스크립트 | 명령어 | 용도 |
|---------|-------|------|
| `arm.py` | `python -m ik_demo.arm` | IK/Ruckig 검증 (헤드리스) |
| `arm.py` | `python -m ik_demo.arm --robot` | 전체 ARM 움직임 검증 |
| `suction.py` | `python -m ik_demo.suction` | Pick/Place 테스트 |
| `gripper.py` | `python -m ik_demo.gripper` | 우측 그리퍼 테스트 |
| `box_pick.py` | `python -m ik_demo.box_pick [--dry]` | 우측 그리퍼 박스 픽 (감지값 대신 손입력 포즈) |
| `move_chassis.py` | `python -m ik_demo.move_chassis` | 샤시 좌우 이동 |
| `jog_ee.py` | `python -m ik_demo.jog_ee` | EE 수동 조그 (torso/EE 값 출력 + x y z 입력 이동, 포즈 따기용) |
| `sequence.py` | `python -m ik_demo.sequence` | 전체 시퀀스 실행 |
| `chassis_sequence.py` | `python -m ik_demo.chassis_sequence` | 샤시+시퀀스 통합 — 시작 시 작업 메뉴 (1 이재 / 2 bin 뚜껑 / 3 box 버리기) |
| `chassis_sequence.py` | `python -m ik_demo.chassis_sequence --box` | 메뉴 3 바로 실행: 우측 그리퍼로 상자 잡고 든 채 수동 이동 → 하역 지점 오른쪽 앞에 내려놓기 |

## 상세 설명

### arm.py — IK/Ruckig 핵심

**헤드리스** (로봇 없음):
```bash
python -m ik_demo.arm
```
- 피노키오 모델 로드
- 6개 교시 포즈 IK 솔브 검증
- Ruckig 궤적 생성
- 따뜻한 시작 성능 벤치마크 (0.82ms/solve)

**로봇 검증**:
```bash
python -m ik_demo.arm --robot
```
- HOME → 6개 포즈 순회 이동
- 분기 전환, 흔들림 없음 확인

---

### suction.py — Pick/Place

```bash
python -m ik_demo.suction
```
- HOME → CASE_PICK로 이동
- Pick 실행 (접촉 감지 + 밀봉)
- CASE_PLACE_R로 이동
- Place 실행

**검증됨**: 접촉 감지 11.4N, 밀봉 성공 ✓

---

### gripper.py — Robotiq 그리퍼

```bash
python -m ik_demo.gripper
```
- Robotiq Modbus 통신 검증 (팔 EE 패스스루 → USB 어댑터 순으로 자동 시도, `ROBOTIQ_TRANSPORT`)
- 개폐 사이클 테스트
- Force feedback 읽기

---

### box_pick.py — 우측 그리퍼 박스 픽 (감지 스텁)

```bash
python -m ik_demo.box_pick --dry                          # 계획만 (로봇 없음)
python -m ik_demo.box_pick                                # 기본 포즈에서 허공 테스트
python -m ik_demo.box_pick --x 0.70 --y -0.35 --top-z 0.60 --yaw-deg 0
python -m ik_demo.box_pick --keep                         # 끝나고 hover에 멈춤 (그립 확인)
python -m ik_demo.box_pick --detect --box-long-m 0.62 --dry  # 헤드 카메라 감지 + 계획만 (팔 안 움직임)
python -m ik_demo.box_pick --detect                          # 감지 + 오른쪽 벽 집기 (잡고 → 10cm 들고 → 내려놓고 → 놓고 → 홈)
python -m ik_demo.box_pick --detect --home-left              # 왼팔 먼저 홈
python -m ik_demo.box_pick --detect --box-long-m 0.62 --carry  # 실제로 들어올리기 (chassis_sequence --box 의 집기 단계와 동일)
```
- `--detect`: BEV OBB 모델(case_detection/runs/obb/box)로 상자 중심·크기·yaw 감지 → 로봇 오른쪽 긴 벽 중점을 집음 (`BOX_GRASP_EDGE_INSET_M`)
- `run_box_pick()`이 이 단계 전체(홈 → 감지 → 집기 → 홈)이며, `chassis_sequence --box`(메뉴 3)가 집기 단계에 호출하는 것과 **같은 함수**입니다. 시퀀스 없이 이 단계만 연습할 때 위 명령을 쓰세요
- `--box-long-m`: 상자 긴 변 실측값. BEV 평면(림 높이)을 크기 일치로 역산. 없으면 `--top-z`를 림 높이로 사용
- 감지 결과 BEV 이미지는 `case_detection/out/box_detect_*.png`에 저장 (초록 OBB, 빨간 십자 = 그립 지점)
- 감지 없이: 박스 윗면 중심 (x, y, top_z)과 장축 yaw를 직접 입력
- 홈 → hover(윗면 15 cm 위) → 수직 하강 → 그리퍼 닫기 → 수직 상승 → (기본) 열고 홈
- 밑에 아무것도 없으면 `no_object`로 보고하고 그대로 올라옵니다 (허공 테스트)
- 튠 값: `BOX_FINGER_LENGTH_M`(베이스→손끝 실측), `BOX_GRASP_YAW_OFFSET_RAD`, `BOX_GRASP_DEPTH_M`, `BOX_HOVER_HEIGHT_M`

---

### move_chassis.py — 샤시 좌우 이동

**대화형 모드** (추천):
```bash
python -m ik_demo.move_chassis
```
명령어:
- `l` — 왼쪽 스트래프 (기본값)
- `r` — 오른쪽 스트래프 (기본값)
- `l 1.0` — 왼쪽 1m
- `r 0.5 0.15` — 오른쪽 0.5m @ 0.15 m/s
- `q` — 종료

**직접 명령**:
```bash
python -m ik_demo.move_chassis --left 1.0
python -m ik_demo.move_chassis --right 0.5 --speed 0.15
```

**설정** (config.py):
```python
CHASSIS_STRAFE_SPEED_MS = 0.1       # m/s
CHASSIS_STRAFE_TIME_S = 7.2         # 기본 시간
CHASSIS_SETTLE_S = 1.0              # 안정화 대기
```

---

### sequence.py — Forward 시퀀스

```bash
python -m ik_demo.sequence
```

순서:
1. Case: CASE_PICK → CASE_PLACE_R
2. Battery 1: BAT_SRC_1 → BAT_SLOT_1
3. Battery 2: BAT_SRC_2 → BAT_SLOT_2

재시도: 설정 `MAX_PHASE_ATTEMPTS`

---

### chassis_sequence.py — 샤시+시퀀스

```bash
python -m ik_demo.chassis_sequence
```

**작업 메뉴** (로봇 연결·양팔 홈 후 표시, 작업 하나 끝나면 다시 메뉴로 돌아옴, `q` 로 종료):
1. case + battery 이재 → 소스 레이어 수(실제 쌓인 개수) / 타겟 레이어 수 / 마지막 case→bin 여부를 물음 (Enter = 기본값 3 / 1 / 예). case+battery 반복은 소스 − 1 회 (3 → 2회), 그 뒤 맨 아래 case 를 bin 으로
2. bin 뚜껑 버리기 (`--lid` 와 동일): 상자 앞에서 `d` → 뚜껑 집기 → (필요하면 물러난 뒤) `d` → 토르소 기울임 + 팔 stow → 하차 지점으로 수동 이동 후 `d` → 바닥 뚜껑 위에 놓기
3. box 버리기 (`--box` 와 동일): 상자 앞에서 `d` → 잡고 듦 → 하역 지점으로 수동 이동 후 `d` → 오른쪽 앞에 내려놓고 홈
4. case 1개 → bin (`--case-bin` 과 동일): 샤시는 외부에서 위치시킴, 자체 strafe 없음 (팔이 안 닿을 때만 닿는 가장 가까운 위치로 최소 이동). case 앞에서 `d` → 감지(1 layer)·픽·파킹 → bin 앞에서 `d` → **bin 감지** 기반으로 내려놓고 홈

1번의 마지막 case → bin 단계도 bin 안의 case가 아니라 **bin 자체를 감지**해서 놓습니다 (bin 안 case 수 `FINAL_BIN_CASE_LAYERS` 는 놓는 높이 계산에만 쓰임).

`--lid` / `--box` / `--box-lid` / `--case-bin` 플래그를 주면 메뉴 없이 바로 그 작업으로 감. 바코드 분류(구 `--gripper`)와 자동 샤시 이동(구 `--auto-move`)은 항상 켜져 있음. `--dashboard`, `--state-publish` 는 그대로 옵션.

**레이어 루프**: 소스 스택이 소진될 때까지 반복. 레이어마다 (case + battery 1/2) 실행 후 스택 높이 자동 갱신 (source −1, target +1) — BEV warp plane이 실제 top face를 따라감.

각 아이템마다:
1. 왼쪽 스트래프 + 감지 + Pick
2. 오른쪽 스트래프 + 감지 + Place
3. 왼쪽 스트래프 (다음 아이템용)

**설정** (config.py): `SRC_LAYERS_REMAINING` / `TGT_LAYERS_REMAINING`은 **시작 스택 높이**만 지정 (실행 시작 시 물리 스택에 맞게 설정). 중단 시 재개용 값이 로그에 출력됨.

---

## 빌드 순서 (권장)

```
1. arm.py (헤드리스)          → IK 작동
2. arm.py --robot             → ARM 움직임 검증
3. move_chassis.py            → 샤시 좌우 이동 테스트
4. suction.py                 → Pick/Place 테스트
5. sequence.py                → 전체 시퀀스
6. chassis_sequence.py        → 최종 통합
```

---

## 설정 (config.py)

모든 파라미터는 `config.py` 한곳에서 관리:
- IK 파라미터
- 동역학 예산 (속도/가속/저크)
- 흡입 강제 임계값
- **Chassis 스트래프 설정**
- 교시 포즈
- 타겟 높이 등

---

**마지막 업데이트:** 2026-07-03
