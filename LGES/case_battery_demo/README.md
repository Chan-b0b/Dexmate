_execute() called for "battery_1" / "battery_2"
│
├─ scan_this = True  (label starts with "battery" AND gripper is wired up)
│
├─ BackgroundScanner.start()          ← fires T triggers in a daemon thread
│
├─ mover.pick()                       ← suction descends; scanner collects reads concurrently
│
├─ BackgroundScanner.stop()
│
├─ mover.lift() + mover.move_to(dst)  ← both branches do this
│
├─ code = scanner.result()            ← agreed value if ≥ BCR_MIN_READS all equal, else None
│
├─ code in cfg.TARGET_BARCODES?
│       │
│      YES → _handoff_to_gripper()   ← grip, suction off, EE sequence, place
│       │         └─ returns False?  → fall through to normal suction place
│       │
│      NO / None → mover.place(dst)  ← original suction-into-case behaviour


# Terminal 1 — services
./run_dashboard_demo.sh

# Terminal 2 — demo (with or without dashboard data)
python -m case_battery_demo.run_demo           # no dashboard
python -m case_battery_demo.run_demo --dashboard  # with live viewer
