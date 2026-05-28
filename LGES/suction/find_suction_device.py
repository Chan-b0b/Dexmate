#!/usr/bin/env python3
"""
Suction Device IP Finder
Scans local networks to find the suction control device
"""

import socket
import requests
import subprocess
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Configuration
NETWORKS = [
    "192.168.1.0/24",      # Original network
    "192.168.50.0/24",     # Current Ethernet network
    "192.168.123.0/24",    # Current WiFi network
]

PORTS_TO_TEST = [80, 502, 5000, 8080, 8000, 3000, 9000, 8888]
TIMEOUT = 2

def test_port(host, port):
    """Test if a specific port is open on a host"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def test_suction_api(host):
    """Test if the host has the suction API endpoint"""
    endpoints = [
        "http://{}:80/api/dc/weblogic/stop",
        "http://{}:8080/api/dc/weblogic/stop",
        "http://{}:5000/api/dc/weblogic/stop",
        "http://{}:3000/api/dc/weblogic/stop",
    ]
    
    for endpoint in endpoints:
        try:
            url = endpoint.format(host)
            response = requests.post(url, timeout=TIMEOUT)
            if response.status_code in [200, 404, 405]:  # Any response means the API exists
                return True, url
        except requests.exceptions.ConnectionError:
            continue
        except Exception:
            continue
    
    return False, None

def scan_host(host):
    """Scan a single host for open ports and suction API"""
    open_ports = []
    
    # Test all ports
    for port in PORTS_TO_TEST:
        if test_port(str(host), port):
            open_ports.append(port)
    
    # If any ports are open, test for API
    api_found = False
    api_url = None
    if open_ports:
        api_found, api_url = test_suction_api(str(host))
    
    if open_ports or api_found:
        return {
            "host": str(host),
            "open_ports": open_ports,
            "api_found": api_found,
            "api_url": api_url
        }
    
    return None

def main():
    print("🔍 Scanning for Suction Device...")
    print(f"🌐 Networks: {', '.join(NETWORKS)}")
    print(f"⏱️  Timeout: {TIMEOUT}s per host\n")
    
    all_ips = []
    for network in NETWORKS:
        try:
            net = ipaddress.ip_network(network)
            all_ips.extend(net.hosts())
        except ValueError as e:
            print(f"❌ Invalid network {network}: {e}")
    
    print(f"📊 Testing {len(all_ips)} hosts...\n")
    
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(scan_host, ip): ip for ip in all_ips}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                print(f"   Progress: {completed}/{len(all_ips)}")
            
            result = future.result()
            if result:
                results.append(result)
    
    # Display results
    print("\n" + "="*70)
    if results:
        print(f"✅ FOUND {len(results)} DEVICE(S):\n")
        for i, result in enumerate(results, 1):
            print(f"Device #{i}:")
            print(f"  Host: {result['host']}")
            print(f"  Open Ports: {result['open_ports']}")
            if result['api_found']:
                print(f"  ✅ Suction API Found at: {result['api_url']}")
            print()
    else:
        print("❌ No devices with open ports found")
        print("   Make sure the suction device is powered on and connected")
        print("   Check if it's on a different network\n")
    
    # Also try to get info from ARP table
    print("="*70)
    print("\n📋 Devices in ARP Table:")
    try:
        arp_output = subprocess.check_output(['arp', '-a']).decode()
        for line in arp_output.split('\n'):
            if '192.168' in line:
                print(f"  {line}")
    except Exception as e:
        print(f"  Could not read ARP table: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("\n💡 If you found the IP, update these files:")
    print("  • /home/dexmate/Dexmate/LGES/suction/test_suction.py")
    print("  • /home/dexmate/Dexmate/LGES/suction/test_suction_gui.py")
    print("  • /home/dexmate/Dexmate/LGES/suction/utils.py")
    print("  • /home/dexmate/Dexmate/LGES/battery_pick/config.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Scan interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
