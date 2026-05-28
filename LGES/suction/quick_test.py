#!/usr/bin/env python3
"""
Quick Suction Device Test - tests known devices
"""

import socket
import subprocess
import sys

DEVICES = [
    "192.168.123.28",
    "192.168.123.80", 
    "192.168.50.21"
]

PORTS = [80, 502, 5000, 8080, 8000, 3000, 9000, 8888]

def test_port(host, port):
    """Test if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def get_hostname(ip):
    """Try to resolve hostname"""
    try:
        return socket.gethostbyaddr(ip)[0]
    except:
        return None

print("🔍 Testing Known Devices for Suction API\n")
print("=" * 60)

for device in DEVICES:
    print(f"\n📍 Testing {device}")
    
    # Try ping first
    try:
        subprocess.run(
            ["ping", "-c", "1", "-W", "1", device],
            capture_output=True,
            timeout=3
        )
        print(f"   ✓ Device is reachable (ping OK)")
    except:
        print(f"   ✗ Device not responding to ping")
        continue
    
    # Test hostname
    hostname = get_hostname(device)
    if hostname and hostname != device:
        print(f"   Hostname: {hostname}")
    
    # Test ports
    open_ports = []
    for port in PORTS:
        if test_port(device, port):
            open_ports.append(port)
    
    if open_ports:
        print(f"   Open Ports: {open_ports}")
        
        # Try HTTP on open ports
        for port in open_ports:
            if port in [80, 8080, 3000, 8000, 8888]:
                try:
                    result = subprocess.run(
                        ["curl", "-s", "-m", "1", f"http://{device}:{port}/api/dc/weblogic/stop"],
                        capture_output=True,
                        timeout=3
                    )
                    if result.returncode == 0:
                        print(f"   ✓ HTTP API found on port {port}")
                        print(f"     → http://{device}:{port}/api/dc/weblogic")
                except:
                    pass
    else:
        print(f"   ⚠️  No common ports open")

print("\n" + "=" * 60)
print("\n💡 Next Steps:")
print("   1. Ensure the suction device is powered on")
print("   2. Check which IP shows open ports")
print("   3. Update config files with that IP")
print("   4. Or, check the device's display/manual for its IP")
