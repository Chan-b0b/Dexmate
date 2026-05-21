# Battery Sorting Control Paradigm

## Overview

Automated battery sorting system using a robot arm with suction end-effector.
Batteries are picked from a source box, barcode-scanned, and stacked into designated target boxes.

---

## Module 1 — Perception: Battery Detection

- **Input:** RGB-D camera frame
- **Output:** 3D pose (position + orientation) of each battery in the source box
- Detects battery bounding boxes, estimates grasp-ready top-face center using depth
- Tracks stacking state (which batteries remain, layer height)

---

## Module 2 — Barcode Scanning

- **Input:** Battery pose from Module 1
- **Output:** Battery ID / sort-class decision
- Moves arm to a fixed "scan pose" above each battery, triggers barcode reader (camera or dedicated scanner)
- Queries lookup table / external DB to decide target box assignment
- Rejects unreadable barcodes (retry or skip logic)

---

## Module 3 — Suction Grasp Execution

- **Input:** 3D grasp pose from Module 1
- **Output:** Grasp success/failure signal
- Runs IK to approach pose → descend → activate suction
- Reads vacuum pressure feedback to confirm seal
- Retries or re-plans on failure

---

## Module 4 — Motion Planning & Trajectory Execution

- **Input:** Current arm state, grasp pose, target place pose
- **Output:** Executed joint trajectory
- Gross motion (pick → lift → transport → place) handled by IK-based trajectory
- Contact phases (approach, place-down) handled by MPC (see Module 4b)
- Uses collision-free path generation

### Module 4b — MPC Contact Controller

Applied at two critical sub-phases:

| Phase | Role |
|---|---|
| Grasp approach | Controlled descent onto battery with bounded contact force — prevents slamming suction cup |
| Place descent | Smooth, force-limited placement into target box slot |

**MPC formulation (receding horizon):**
- State: joint positions, velocities, end-effector wrench
- Control input: joint torques / velocity commands
- Constraints: joint limits, velocity/torque limits, max contact force
- Cost: tracking error + control effort + force smoothness
- Horizon: ~500 ms, re-solved every control cycle (~10–50 ms)

**Practical split:**
- IK handles gross motion (fast, low compute)
- MPC activates only during contact phases (bounded, predictable duration)

---

## Module 5 — Place & Stack Management

- **Input:** Battery ID → target box assignment
- **Output:** Place pose, updated stack state
- Maintains a stack register per box (row, column, layer)
- Computes next available slot pose in target box
- Deactivates suction at place, confirms drop with vacuum release signal

---

## Module 6 — Task Orchestrator (State Machine)

```
IDLE → DETECT → SCAN → GRASP → TRANSPORT → PLACE → [back to DETECT]
```

Error states handled at each transition:
- Detection fail → retry or skip
- Scan fail → retry N times → reject battery
- Grasp drop (vacuum lost mid-transport) → re-pick
- Box full → pause and alert operator

---

## Module 7 — Configuration & Box Layout

- Source/target box geometries (dimensions, slot grid)
- Camera-to-robot extrinsic calibration parameters
- Barcode-to-sortclass mapping table
- Suction approach offsets per battery model
- MPC tuning parameters (horizon length, cost weights, force limits)

---

## Suggested Build Order

```
Config (7) → Perception (1) → Barcode (2) → Suction Grasp (3)
→ IK Motion (4) → MPC Contact (4b) → Place Manager (5) → Orchestrator (6)
```

---

## Control Architecture Summary

```
         ┌─────────────┐
         │ Orchestrator │  (Module 6)
         └──────┬───────┘
     ┌──────────┼──────────┐
     ▼          ▼          ▼
 Perception  Barcode   Stack Mgr
 (Mod 1)    (Mod 2)   (Mod 5)
                │
          Grasp Exec (Mod 3)
                │
     ┌──────────┴──────────┐
     ▼                     ▼
  IK Trajectory        MPC Contact
  (gross motion)       (approach/place)
     └──────────┬──────────┘
                ▼
          Robot Hardware
```
