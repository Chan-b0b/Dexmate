#!/usr/bin/env python3
"""
Suction Device Cable Detection
Monitor network changes when unplugging/replugging the LAN cable
"""

import subprocess
import time
import sys
from collections import defaultdict

def get_arp_table():
    """Get current ARP table"""
    try:
        output = subprocess.check_output(['arp', '-a'], universal_newlines=True)
        devices = {}
        for line in output.split('\n'):
            if '192.168' in line:
                parts = line.split()
                if len(parts) >= 3:
                    ip = parts[1].strip('()')
                    mac = parts[3]
                    devices[ip] = mac
        return devices
    except Exception as e:
        print(f"Error reading ARP: {e}")
        return {}

def get_interface_status():
    """Get status of network interfaces"""
    try:
        output = subprocess.check_output(['ip', 'addr'], universal_newlines=True)
        interfaces = {}
        for line in output.split('\n'):
            if '192.168' in line and 'inet ' in line:
                parts = line.split()
                ip = parts[1].split('/')[0]
                interfaces[ip] = True
        return interfaces
    except Exception as e:
        print(f"Error reading interfaces: {e}")
        return {}

def format_table(devices):
    """Pretty print device table"""
    print("  IP Address        MAC Address")
    print("  " + "-" * 45)
    for ip, mac in sorted(devices.items()):
        print(f"  {ip:17} {mac}")

def main():
    print("=" * 70)
    print("🔌 Suction Device Cable Detection Tool")
    print("=" * 70)
    print()
    print("This tool will help identify the suction device by detecting")
    print("network changes when you unplug/replug the LAN cable.")
    print()
    
    # Get baseline
    print("📸 Taking baseline snapshot of current network...")
    time.sleep(1)
    baseline_arp = get_arp_table()
    baseline_interfaces = get_interface_status()
    
    print("\n📋 Current devices on network:")
    if baseline_arp:
        format_table(baseline_arp)
    else:
        print("  (No 192.168.x.x devices found)")
    
    print("\n" + "=" * 70)
    print("👉 NOW: Unplug the suction device's LAN cable")
    print("=" * 70)
    print()
    
    input("Press ENTER once you've unplugged the cable...")
    
    print("\n⏳ Waiting for network to stabilize (3 seconds)...")
    time.sleep(3)
    
    # After unplugging
    print("\n📸 Checking network after unplugging...")
    after_unplug_arp = get_arp_table()
    
    # Find what disappeared
    disappeared = {}
    for ip, mac in baseline_arp.items():
        if ip not in after_unplug_arp:
            disappeared[ip] = mac
    
    if disappeared:
        print("\n✅ DEVICES THAT DISAPPEARED:")
        format_table(disappeared)
        print("\n⚠️  These are candidates for the suction device!")
    else:
        print("\n❌ No devices disappeared. The cable might not have been connected,")
        print("   or the device has DHCP enabled and reconnected too quickly.")
    
    print("\n" + "=" * 70)
    print("👉 NOW: Plug the suction device's LAN cable back in")
    print("=" * 70)
    print()
    
    input("Press ENTER once you've plugged the cable back in...")
    
    print("\n⏳ Waiting for network to stabilize (3 seconds)...")
    time.sleep(3)
    
    # After replugging
    print("\n📸 Checking network after replugging...")
    after_replug_arp = get_arp_table()
    
    # Find what reappeared
    reappeared = {}
    for ip, mac in after_replug_arp.items():
        if ip not in after_unplug_arp:
            reappeared[ip] = mac
    
    if reappeared:
        print("\n✅ DEVICES THAT REAPPEARED:")
        format_table(reappeared)
        print("\n🎯 This is your SUCTION DEVICE!")
    else:
        print("\n❌ No devices reappeared. Device may not have reconnected yet.")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print()
    
    if disappeared and reappeared:
        for ip, mac in reappeared.items():
            print(f"✅ Suction Device IP: {ip}")
            print(f"   MAC Address: {mac}")
            print()
            print("Next steps:")
            print(f"  1. Use this IP to update the configuration:")
            print(f"     python3 update_config.py {ip}")
            print()
            print("  2. Or manually update these files:")
            print("     • LGES/suction/test_suction.py")
            print("     • LGES/suction/test_suction_gui.py")
            print("     • LGES/suction/utils.py")
            print("     • LGES/battery_pick/config.py")
    else:
        print("❌ Could not determine suction device")
        print()
        print("Troubleshooting tips:")
        print("  1. Make sure you're unplugging the suction device, not router/other")
        print("  2. Wait for the unplugged message before pressing Enter")
        print("  3. Try again - sometimes DHCP takes a moment")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
