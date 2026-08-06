# Live authority probe 런 정리 — fromnaive vs naive (2026-08-04 ~ 08-06)

모든 런 공통:
- **film 모델**: `Chanho-Lee/smolvla_film_0729_prefix_mask1_recal_fromnaive` main = **val-best@2,500**
  (naive best@10k에서 warm-start, 20k 학습, recal 캘리브레이션 F0/FMAG 5.5/1, FZ 3.0/0.7)
- **naive 베이스라인**: `Chanho-Lee/smolvla_naive_0729` main = val-best@10k — **같은 동결 관측**에 예측
- 도구: `probe_film_authority_live.py` (예측만, 액션 미전송). 정량 판독: `EVIDENCE.md` §3.7,
  도스 비대칭·재앵커 논의: DISCUSSION_LOG 08-04~06.

| 폴더 | 날짜 | 상태 | 내용 |
|---|---|---|---|
| `run1_0804_stale-env-offsets_FILM-INVALID` | 08-04 14:33 | **film 무효 / naive 유효** | env 파일 오프셋(13:49 수동 측정, 6.53/3.35)이 probe 포즈 실측(4.9N/0.9N) 대비 ~0.7N 과보정 → c-hat 오앵커. naive 쪽(st_fc +1.58, st_sealed +3.37)만 인용 가능. 오프셋 포즈 민감성의 실증 사례. |
| `run2_0806_hover-reanchor_undosed-swaps_PARTIAL` | 08-06 15:15 | **부분 유효** | 하강 전 hover 자체 재앵커(`--baseline-hover`) 도입 첫 런. 단 절대값 swap이 드리프트만큼 저도스로 읽혀(film fc가 preseal 수준) film-vs-naive 비교는 비대칭. film c-hat 앵커 시나리오(hover→preseal→sealed 단조)와 naive 쪽은 유효. |
| `run3_0806_fair-dose_10poses_VALID` | 08-06 15:24 | **유효** | swap 드리프트 보정(`swap_drift` JSON 기록) — 두 모델이 같은 물리 반사실 수령. 중간 도스(≤9N): naive ≥ film (오프라인 곡선 같은 구간과 정합). film 도스-반응 단조: hover +0.29 → preseal +0.87 → sealed +1.41. |
| `run4_0806_high-dose_fz6-9-12N_3poses_VALID` | 08-06 15:29 | **유효 — 핵심 결과** | 고도스 스윕 fz +6/9/12N, 3포즈: **crossover 온로봇 재현** — film +1.19→+4.59→+8.56(가속·후퇴 진입) vs naive +1.34→+3.18→+4.18(한계반응 붕괴). 오프라인 기울기 대비의 실기 확인. |

주의: run2까지의 `st_firstcontact/st_sealed` film 수치는 도스 비대칭 때문에 naive와 직접 비교
금지. 논문 인용은 run3(공정 도스)·run4(고도스)만, run1은 "측정-포즈 민감성" 운영 증거로만.
