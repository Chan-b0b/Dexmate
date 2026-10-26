# pick_place → chassis_sequence 이식 목록

pick_place에서 먼저 바꾸고 로봇에서 확인한 뒤, chassis_sequence에 옮길 것들.
줄 번호는 2026-10-12 기준.

## 이식 상태 (2026-10-12, chassis_sequence 로봇 미검증)

**옮김**: A1·A2·A3·A4·A5(misseat 판정·복구 완전히 끔)·A6·A7(`SEED_BIN_CENTER_OFFSET`는 유지)·A8·A9·A10(`PLACE_RELEASE_ABOVE_M`, 기본 None)·B1·B2·B3·B6.
- 공통 코드는 suction.py / config로: `pick(planned=True)`, `place(straight=True, hover_m=, release_above_m=)`,
  `_descend_straight`, `_tip_snapshot`, `_lift_to_transport`의 `LIFT_VERTICAL_TO_Z` 분할(전체 적용).
  planned/straight는 **켜는 인자**라 lid·sequence.py·pick_check·case_to_bin 등 다른 호출부는 그대로.
- B3 높은 테이블 값은 기본 config에 반영(`TORSO_JOINTS`, `SOURCE_CASE_CENTER`, `DIVERT_BIN_PLANE_Z_M`,
  `BOX_RIM_Z_M`, `SAFE_TRANSPORT_Z`, `LIFT_CLEAR_EE_Z`) — lid/box 작업에도 적용됨.
- 배터리 기준 높이(A9)는 chassis_sequence에서 **case item에만**: source는 `src_layers >= 2`(맨 아래 case 제외),
  target은 seed가 아닐 때. 추가 안전장치: 배터리 윗면이 `top_face_z(layers + 1)` ±`BATTERY_TOP_GATE_M`(10mm)
  밖이면 실패 → operator. 실패는 모두 operator(사용자 결정).
- B6 창은 **모든 모드**에서 source·target·seed bin·bin place·divert에 적용(사용자: 창이 기존 정렬을 대체).
  chassis_sequence에서 `_center_case`(yaw 회전·item별 기준점·leg 학습), `_auto_adjust`(도달 실패 시 최소 이동),
  seed의 `_align_to_bin` 호출을 없앰 → 도달 실패는 바로 operator. `_auto_adjust`·`_column_quiet`는 삭제,
  `_center_case`·`_align_to_bin`은 case_to_bin.py가 써서 남김. `run_item(strict_start=)` 인자 삭제.
- 창 이동의 좌우 이동량이 이동 거리 학습(`ChassisNav.learn`)에 들어감: 도착당 첫 이동만, `skip_next_learn` 존중.
  돌아오는 이동의 item별 기준점 미리 더하기(`next_y_ref`/`next_pose_key`) 삭제.
  학습값은 `run_logs/chassis_legs.json`에 저장돼 다음 실행이 이어씀(config 값이 바뀌면 무시, 범위는 config ±20cm).
- 배터리 윗면 추정: chassis_sequence만 **칸 영역 p90 − 3mm**로 바꿈(배터리 1개일 때 흔들림 13.5→3.4mm). pick_place는 아직 옛 덩어리 규칙.
  target 안전장치 기준 층 = 층 카운터 − 1(1층 case가 seed bin으로 가도 카운터는 올라감) — 버그 수정.
- 상수는 config/chassis.py 끝(`OBB_DEPTH_*`, `BATTERY_TOP_*`, `PLACE_RELEASE_*`, `WINDOW_*`).
- (1012 오후) view park: `ARM_VIEW_PARK_EE_POS` (0.99, 0.40, 1.05) → (0.80, 0.40, 1.17) 기본 config에 —
  높은 테이블 TORSO에서 옛 점이 62mm 모자라 joint park로 빠졌음(pick_place 6/6). pick_place override와
  `_view_park(park=)` 인자는 지움. z 1.17 로봇 미검증(1.12는 pick_place에서 확인).
- (1012 오후) 빈 bin 놓기를 bin 방향에 맞춤(`_detect_bin_xy(with_yaw=True)`, 0/180 경계는 프레임별
  [-90, 90) 변환 후 중앙값) + manual 모드면 놓을 중심을 `_into_window`로: `_place_case_in_bin`과
  run_item의 **seed place** 둘 다(seed는 원래 source yaw로 놓았음). 로봇 미검증.
- 빈 bin 바닥은 depth로 재지 않음: 주황 bin 바닥을 ZED가 0.85~0.88(+170~200mm)로 읽음(1012 13:33, 13:43).
  설정값(평면 0.6838, 예상 ee z 0.8338)은 1012 석션 접촉으로 확인(bin 테이블 touch ee 0.8340).

**안 옮김**: B5(pick_place 전용 반복 입력). B4는 B3 반영으로 불필요. pick_place는 손대지 않음 —
place_config와 config에 같은 상수가 두 벌 있음(나중에 pick_place가 공통 함수를 쓰게 정리 가능).
case_to_bin.py는 여전히 corner seat.

## A. 1012 — 비전만으로 한 번에 놓기 (벽 밀기 제거)

pick_place 현재 상태: pose = detection 결과 + case 기준 offset(`CASE_GRAB_OFFSET`,
`SLOT_OFFSETS`)만. trim·corner bias·옆으로 미는 동작 모두 없음. **로봇 미검증.**

| # | chassis_sequence 위치 | 지금 하는 것 | 옮길 것 |
|---|---|---|---|
| A1 | `run_item` target 분기 [chassis_sequence.py:3125-3160] | `PLACE_X_LAYER_TRIM_M`(현재 0), `PLACE_X_PLANE_TRIM_M` −10mm, `PLACE_YAW_TRIM_RAD`(battery), corner aim bias(case 30mm / battery 10mm) | 전부 제거 |
| A2 | 같은 곳, `intended[0] += ...` [3129, 3136] | 놓은 뒤 검증할 때 쓰는 "의도한 위치"에 trim을 더함 | A1과 같이 제거 (검증이 detection 기준으로 착지 오차를 재게) |
| A3 | `run_item` place 호출 [3206-3208] | `corner_seat="case"/"battery"` | `corner_seat=None` + battery는 `hover_m=BATTERY_PLACE_HOVER_M` |
| A4 | `SuctionMover.place` (suction.py) | `corner_seat=None`이면 hover→creep line을 계획 경로로 한 번에 내려감 (0909 battery 흔들림 구간) | place_mover의 `_descend_straight` + `hover_m` 인자를 suction.py로 옮김 |
| A5 | `run_item`이 넘기는 `misseat_tol_m=mtol` [2975, 3061] | corner seat를 끄면 `_misseat_recover`가 켜짐(`PLACE_RECOVER_ATTEMPTS`=6, 힘을 따라 xy를 옮기며 다시 앉히는 동작) — 이것도 "비전 말고 다른 걸로 위치 맞추기" | 끌지 / 판정만 하고 operator gate로 넘길지 결정 필요 (pick_place는 `misseat_tol_m`을 안 넘겨서 꺼져 있음) |
| A6 | `_divert_case_place` battery [2568-2575, 2610-2614] | `PLACE_YAW_TRIM_RAD` + battery corner bias + `corner_seat="battery"` | A1·A3과 같은 처리 |
| A7 | `_place_case_in_bin` bin [2720-2752] | `SEED_BIN_CENTER_OFFSET` (−11.5, +29.3)mm + case corner bias + `corner_seat="case"` | **미정.** pick_place `--target bin`도 아직 이 함수를 그대로 씀 → pick_place에서 먼저 바꿔서 확인 |
| A8 | 시작 로그 [3718, 3800-3803] | corner bias 값을 출력 | A1 후 정리 |
| A9 | `run_item` source 검출 / target 깊이 측정 [3065-3090] | source: 모델 높이(층 수) 또는 ZTracker 앵커. target: 검출 중심 25px 창 중앙값(`_depth_face_z`), 실패 시 모델 높이 | **로봇 확인 완료(1012, 3→2층·2→1층 제자리 pick/place)** pick_place `--obb-depth`: OBB 안쪽(슬롯 영역) depth의 가장 높은 군집 = 아래 case의 **배터리 윗면**(`obb_plane_z`). xy용 BEV 높이 = 배터리 윗면 − `BATTERY_TOP_ABOVE_FACE_M`(16mm), source(빈 case)는 + 1층. 하강 예상 z = 배터리 윗면 + 컵 길이(빈 case가 배터리 위에 직접 얹힘). 투명 case 면 depth는 프레임마다 ±15mm 흔들려서 못 씀. 실패·8cm 넘는 재검출 점프는 operator로. 상수는 place_config `OBB_DEPTH_*`, `BATTERY_TOP_ABOVE_FACE_M`. 참고: place 접촉이 예상 z보다 보통 8~11mm 낮음 |
| A10 | `run_item` place [3206-3208] | 힘(10N) 감지까지 내려가 놓음 | **실험 중(1012)**: pick_place `PLACE_RELEASE_ABOVE_M`=5mm — 예상 z(depth 평면+컵 길이)+5mm에서 멈추고 prelift 없이 놓음, 그 전 접촉은 `early_contact`로 operator. 로봇 결과 보고 이식 여부 결정 |

이식할 필요가 없는 것:
- `PICK_WALL_RELIEF_M`: pick_place 전용 동작이었고 지금은 0으로 꺼져 있음. suction.py의 pick에는 원래 없음.
- 흡착 재시도 nudge (`PICK_SEAL_RETRY_OFFSET_M`): `retry_dir`를 넘길 때만 옆으로 움직임. 양쪽 다 안 넘김.

## B. 1012 이전부터 pick_place에만 있는 차이

| # | 내용 | 메모 |
|---|---|---|
| B1 | `PlaceMover.pick`: transport→hover→creep line까지 **계획 경로**, 거기서 creep | suction.py `pick`은 반대로 hover부터 **매 tick 하강**. 두 docstring이 서로 상대 방식이 나빴다고 적고 있음(suction: 0907 계획 경로 65mm 흔들림 / place_mover: 매 tick 방식이 28mm 앞에서 잡음). 이식 전에 로봇 로그로 어느 쪽이 맞는지 정해야 함 |
| B2 | `_lift_to_transport`: `LIFT_VERTICAL_TO_Z`(1.0)까지만 수직 per-tick, 나머지는 계획 경로 | per-tick 상승이 고속에서 늦게 따라감(피치 −10°, 컵 끝 80mm) |
| B3 | `place_config.OVERRIDES` (테이블이 높아짐, 바닥 0.60→0.67): `TORSO_JOINTS`, `SOURCE_CASE_CENTER` z 0.80→0.87, `DIVERT_BIN_PLANE_Z_M` 0.6138→0.6838, `BOX_RIM_Z_M` 0.65→0.805(바닥 0.67 + `BOX_WALL_HEIGHT_M` 0.135), `SAFE_TRANSPORT_Z` 1.1→1.17, `LIFT_CLEAR_EE_Z` 1.05→1.12 | 높이 값은 이전 값 +0.07로 **정한 것이지 측정값이 아님** — 로봇 미검증, 특히 `SAFE_TRANSPORT_Z`는 IK 도달 범위 확인 필요. chassis_sequence를 높은 테이블에서 돌리려면 config에 반영. `pin_torso` override는 이 TORSO 값을 쓰려고 있는 것(arm.py가 config를 직접 읽어서 override가 안 닿았음). `chassis_sequence`가 config를 직접 읽는 값(`SOURCE_CASE_CENTER`, `DIVERT_BIN_PLANE_Z_M`)은 pick_place에서도 override가 안 닿을 수 있음 |
| B4 | `place_in_bin`이 `cs.cfg`의 `DIVERT_BIN_PLANE_Z_M`, `SEED_BIN_CENTER_OFFSET`를 place_config 값으로 덮어씀 | B3이 config에 들어가면 불필요 |

## B-2. 1012 이후 pick_place 운영 방식 (로봇 미검증)

| # | 내용 | 메모 |
|---|---|---|
| B5 | **pick→place 반복**: place 성공 후 `safe_home` → 다음 입력(`Enter`=같은 설정, `q`=종료, `battery1 3 2` / `3 2` / `case 3 2` = item, src 층수, tgt 층수 변경) → 머리 24° 복귀 → source 스테이션부터 다시 | `_ask_next`, `main`의 `while True`. 층수는 자동으로 줄이거나 늘리지 않고 직접 입력. 실패(detect 실패·도달 불가·`q`)는 물건을 든 채 넘어가지 않도록 바로 종료. `--target bin`은 item을 case로만 |
| B6 | **detect 위치 창**: 다듬은 detect(`_refine_det`) 결과의 x,y가 창(`WINDOW_X_M` 0.68±0.02, `WINDOW_Y_M` 0±0.10) 밖이면 섀시를 창 **중심**(0.68, 0.0)으로 대각선 이동(두 축 동시, 한 축만 밖이어도 두 축 모두 중심으로) → 처음부터 re-detect. 최대 `WINDOW_MAX_MOVES`(4)번, 그래도 밖이면 `_manual_strafe`로 operator에게 넘김 | source pick 전(`pick_at_source`)과 target place 전(`place_at_target`) 둘 다. 축당 이동은 `CHASSIS_ADJUST_MAX_TRANSLATE_M`(0.30)로 제한. 창 값 0.68은 pick_place에서 정한 값이고 `SOURCE_CASE_CENTER` x(0.88)와 다름 → 이식 전 어느 스테이션에 맞는지 확인 |
| B7 | `move_chassis.move_diagonal(bot, dx, dy, speed)`: 바퀴를 (dx, dy) 방향으로 미리 꺾고 한 번에 이동 | **`ik_demo/move_chassis.py`에 추가돼 있어 이미 공통**(할 일 없음). 대각선 조향은 처음 쓰는 방향이라 로그의 `pre-steer done:` 확인. 조향 불가 시 앞뒤→좌우 두 번으로 대체 |

## C. 이미 공통 (할 일 없음)

- `FLOOR_Z_BASE_M` 0.60 → 0.67 (측정값, `case_detection/config.py`와 `ik_demo/config/sequence.py` 둘 다): 양쪽 모두 읽음. 이 값에서 계산되는 `DESCENT_CHECK_BOTTOM_EE_Z`, `TARGET_DEFAULT_CASE_CENTER` z, BEV 평면도 따라감. **착지 접촉은 로봇에서 다시 확인 필요**(chassis.py 주석).
- `SUCTION_LENGTH_M` 0.155 → 0.150 (config/geometry.py, 1012 depth_check 측정): 양쪽 모두 이 값을 읽음.
- target 높이를 depth로 재고 다시 검출 (0915): chassis_sequence `run_item`에도 있음 [3065-3090].
