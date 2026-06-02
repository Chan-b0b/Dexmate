#!/usr/bin/env python3
"""Debug script to test VacuumMonitor connection and DI0 events."""

import time
import sys

# Add case_battery_demo to path
sys.path.insert(0, 'case_battery_demo')

from case_battery_demo import suction_io, config as cfg

print("=" * 60)
print("VACUUM SEAL MONITOR DEBUG")
print("=" * 60)
print(f"Suction controller: {cfg.SUCTION_HOST}")
print(f"Suction ON ID: {cfg.SUCTION_ON_ID}")
print(f"Suction OFF ID: {cfg.SUCTION_OFF_ID}")
print()

# Start the vacuum monitor
print("Starting VacuumMonitor...")
vac = suction_io.VacuumMonitor()
vac.start()

# Wait for socketio connection
print("Waiting 3 seconds for socketio connection...")
time.sleep(3.0)

if vac.is_connected():
    print("✓ CONNECTED to suction controller!")
else:
    print("✗ NOT CONNECTED — check network and suction controller IP")

print(f"Initial seal state: {vac.is_sealed()}")

# Turn on suction
print("\n" + "=" * 60)
print("Turning ON suction...")
suction_io.suction_on()

print("Monitoring DI0 for 5 seconds while suction is ON...")
for i in range(10):
    state = vac.is_sealed()
    print(f"  t={i*0.5:.1f}s: DI0 sealed = {state}")
    time.sleep(0.5)

print("\n" + "=" * 60)
print("Testing wait_for_seal with 3s timeout...")
start = time.time()
sealed = vac.wait_for_seal(timeout=3.0)
elapsed = time.time() - start
print(f"  Result: sealed={sealed}, elapsed={elapsed:.2f}s")

# Turn off suction
print("\nTurning OFF suction...")
suction_io.suction_off()

print("Monitoring DI0 for 2 more seconds after OFF...")
for i in range(4):
    state = vac.is_sealed()
    print(f"  t={i*0.5:.1f}s: DI0 sealed = {state}")
    time.sleep(0.5)

vac.stop()
print("\nDebug complete.")
