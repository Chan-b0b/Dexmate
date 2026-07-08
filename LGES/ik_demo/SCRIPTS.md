# ik_demo 실행 스크립트

각 모듈의 실행 방법과 용도를 정리한 간단한 가이드입니다.

## 빠른 참조

| 스크립트 | 명령어 | 용도 |
|---------|-------|------|
| `arm.py` | `python -m ik_demo.arm` | IK/Ruckig 검증 (헤드리스) |
| `arm.py` | `python -m ik_demo.arm --robot` | 전체 ARM 움직임 검증 |
| `suction.py` | `python -m ik_demo.suction` | Pick/Place 테스트 |
| `gripper.py` | `python -m ik_demo.gripper` | 우측 그리퍼 테스트 |
| `move_chassis.py` | `python -m ik_demo.move_chassis` | 샤시 좌우 이동 |
| `sequence.py` | `python -m ik_demo.sequence` | 전체 시퀀스 실행 |
| `chassis_sequence.py` | `python -m ik_demo.chassis_sequence` | 샤시+시퀀스 통합 |

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
- Robotiq Modbus 통신 검증
- 개폐 사이클 테스트
- Force feedback 읽기

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
