#!/usr/bin/env python3
"""Test the new toolA-based seal detection."""

import time
import sys

sys.path.insert(0, 'case_battery_demo')

from case_battery_demo import suction_io, config as cfg

print("=" * 70)
print("TESTING TOOL CURRENT SEAL DETECTION")
print("=" * 70)
print(f"Suction threshold: {cfg.SUCTION_SEAL_CURRENT_A:.4f} A")
print()

vac = suction_io.VacuumMonitor()
vac.start()

print("Waiting for connection...")
time.sleep(3)

if vac.is_connected():
    print("✓ Connected!")
    print()
    
    # Test 1: Suction OFF
    print("TEST 1: Suction OFF (baseline)")
    print("-" * 70)
    for i in range(3):
        current = vac.get_tool_current()
        sealed = vac.is_sealed()
        print(f"  {i*1.0:.1f}s: toolA={current:.4f}A, sealed={sealed}")
        time.sleep(1.0)
    
    # Test 2: Turn suction ON
    print("\nTEST 2: Turning suction ON")
    print("-" * 70)
    suction_io.suction_on()
    
    print("Monitoring seal detection (5 seconds)...")
    for i in range(5):
        current = vac.get_tool_current()
        sealed = vac.is_sealed()
        print(f"  {i*1.0:.1f}s: toolA={current:.4f}A, sealed={sealed}")
        time.sleep(1.0)
    
    # Test 3: Turn suction OFF
    print("\nTEST 3: Turning suction OFF")
    print("-" * 70)
    suction_io.suction_off()
    
    for i in range(3):
        current = vac.get_tool_current()
        sealed = vac.is_sealed()
        print(f"  {i*1.0:.1f}s: toolA={current:.4f}A, sealed={sealed}")
        time.sleep(1.0)
    
    vac.stop()
    
    print("\n" + "=" * 70)
    print("✓ TEST COMPLETE")
    print("=" * 70)
    print("If sealed went True when suction was ON, the detection works!")
    
else:
    print("✗ Not connected!")

