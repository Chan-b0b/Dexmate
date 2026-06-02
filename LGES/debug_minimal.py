#!/usr/bin/env python3
"""Minimal debug to check socketio connection and raw events."""

import time
import sys
import logging

# Set up logging to see everything
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

sys.path.insert(0, 'case_battery_demo')

from case_battery_demo import suction_io, config as cfg

print("=" * 70)
print("SOCKETIO CONNECTION TEST")
print("=" * 70)
print(f"Target: http://{cfg.SUCTION_HOST}")
print()

vac = suction_io.VacuumMonitor()

# Add raw event logging to see what comes in
print("Starting VacuumMonitor...")
vac.start()

print("\nWaiting 4 seconds for connection...")
time.sleep(4.0)

print(f"\n✓ Connected: {vac.is_connected()}")
print(f"✓ Sealed state: {vac.is_sealed()}")

if vac.is_connected():
    print("\n[SUCCESS] Socketio IS connected! Now testing with suction...")
    
    print("\nTurning ON suction (should trigger DI0 events)...")
    suction_io.suction_on()
    
    print("Waiting 3 seconds for DI0 to change...")
    for i in range(6):
        print(f"  {i*0.5:.1f}s: sealed={vac.is_sealed()}")
        time.sleep(0.5)
    
    print("\nTurning OFF suction...")
    suction_io.suction_off()
    time.sleep(1.0)
else:
    print("\n[FAILURE] Socketio NOT connected!")
    print("Check:")
    print(f"  1. Suction controller at {cfg.SUCTION_HOST} is online")
    print(f"  2. Network connectivity")
    print(f"  3. Firewall isn't blocking port 80/websocket")

vac.stop()
print("\nDone.")
