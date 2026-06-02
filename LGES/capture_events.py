#!/usr/bin/env python3
"""Log all socketio events from suction controller to see data structure."""

import time
import sys
import json
from pprint import pprint

sys.path.insert(0, 'case_battery_demo')

from case_battery_demo import suction_io, config as cfg

print("=" * 70)
print("SUCTION CONTROLLER DATA CAPTURE")
print("=" * 70)
print(f"Target: http://{cfg.SUCTION_HOST}")
print()

class RawVacuumMonitor:
    """Enhanced monitor that captures all socketio events."""
    
    def __init__(self, host: str = cfg.SUCTION_HOST):
        import socketio as _sio_module
        from loguru import logger
        
        self._sio = _sio_module.Client()
        self._host = host
        self._events = []
        self._logger = logger
        
        @self._sio.on("*")
        def _on_data(event, data):
            """Capture ALL events."""
            self._events.append({
                'event': event,
                'data': data,
                'timestamp': time.time()
            })
            print(f"\n[EVENT #{len(self._events)}] {event}")
            print("-" * 70)
            try:
                pprint(data, width=70, compact=False)
            except:
                print(f"  {str(data)[:500]}")
            print()
    
    def connect(self):
        try:
            self._logger.info("[Monitor] Connecting to http://{}/socket.io", self._host)
            self._sio.connect(
                f"http://{self._host}",
                transports=["websocket", "polling"],
                socketio_path="socket.io",
            )
            self._logger.info("[Monitor] Connected!")
            return True
        except Exception as exc:
            self._logger.error("[Monitor] Connection failed: {}", exc)
            return False
    
    def wait(self, duration: float = 3.0):
        """Wait and receive events."""
        print(f"Listening for events ({duration:.1f}s)...")
        time.sleep(duration)
        print(f"Captured {len(self._events)} events")
    
    def disconnect(self):
        try:
            self._sio.disconnect()
        except:
            pass


# Test 1: Capture background events
print("TEST 1: Listening for background events (no suction)")
print("=" * 70)
mon = RawVacuumMonitor()
if mon.connect():
    time.sleep(2)  # Give connection time to stabilize
    mon.wait(3.0)
    mon.disconnect()

# Test 2: Capture events while suction is ON
print("\n\nTEST 2: Listening while suction is ON")
print("=" * 70)
mon2 = RawVacuumMonitor()
if mon2.connect():
    time.sleep(2)
    print("Turning ON suction...")
    suction_io.suction_on()
    mon2.wait(5.0)
    
    print("\nTurning OFF suction...")
    suction_io.suction_off()
    mon2.wait(2.0)
    
    mon2.disconnect()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if mon._events:
    print(f"✓ Received {len(mon._events)} events during idle")
    print("Data keys found:", set(str(e['data'].keys()) if isinstance(e['data'], dict) else type(e['data']) for e in mon._events))
else:
    print("✗ No events received during idle")

if mon2._events:
    print(f"✓ Received {len(mon2._events)} events during suction activity")
    print("Data keys found:", set(str(e['data'].keys()) if isinstance(e['data'], dict) else type(e['data']) for e in mon2._events))
    
    # Look for force/pressure data
    for e in mon2._events:
        data_str = str(e['data']).lower()
        if any(x in data_str for x in ['force', 'pressure', 'pascal', 'bar', 'kpa', 'output', 'value']):
            print(f"\n⚡ Potential force/pressure data in event '{e['event']}':")
            pprint(e['data'], width=70)
else:
    print("✗ No events received during suction activity")

print("\nDone.")
