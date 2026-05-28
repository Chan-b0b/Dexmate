#!/usr/bin/env python3
"""
Update Suction Device IP in all configuration files
"""

import sys
import re
from pathlib import Path

# Files to update and their patterns
CONFIG_FILES = {
    "LGES/suction/test_suction.py": [
        (r'MODBUS_HOST\s*=\s*"[^"]*"', 'MODBUS_HOST = "{ip}"'),
        (r'requests\.post\("http://[^"]*/', 'requests.post("http://{ip}/'),
    ],
    "LGES/suction/test_suction_gui.py": [
        (r'BASE_URL\s*=\s*"http://[^"]*/', 'BASE_URL = "http://{ip}/'),
    ],
    "LGES/suction/utils.py": [
        (r'"http://[^"]*"', '"http://{ip}"'),
    ],
    "LGES/battery_pick/config.py": [
        (r'SUCTION_BASE_URL:\s*str\s*=\s*"http://[^"]*/', 'SUCTION_BASE_URL: str = "http://{ip}/'),
        (r'SUCTION_HOST:\s*str\s*=\s*"[^"]*"', 'SUCTION_HOST: str = "{ip}"'),
    ],
}

def update_files(new_ip, base_dir="."):
    """Update all config files with new IP"""
    base_path = Path(base_dir)
    updated_files = []
    failed_files = []
    
    for file_path, patterns in CONFIG_FILES.items():
        full_path = base_path / file_path
        
        if not full_path.exists():
            print(f"⚠️  File not found: {file_path}")
            failed_files.append(file_path)
            continue
        
        try:
            # Read file
            with open(full_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Apply replacements
            for pattern, replacement in patterns:
                replacement_text = replacement.format(ip=new_ip)
                content = re.sub(pattern, replacement_text, content)
            
            # Only write if changed
            if content != original_content:
                with open(full_path, 'w') as f:
                    f.write(content)
                updated_files.append(file_path)
                print(f"✅ Updated: {file_path}")
            else:
                print(f"⏭️  No changes needed: {file_path}")
        
        except Exception as e:
            print(f"❌ Error updating {file_path}: {e}")
            failed_files.append(file_path)
    
    return updated_files, failed_files

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 update_config.py <IP_ADDRESS>")
        print()
        print("Example:")
        print("  python3 update_config.py 192.168.123.80")
        print()
        print("This will update all config files with the new suction device IP")
        sys.exit(1)
    
    new_ip = sys.argv[1]
    
    # Validate IP format
    if not re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', new_ip):
        print(f"❌ Invalid IP address: {new_ip}")
        sys.exit(1)
    
    print("=" * 70)
    print("🔄 Suction Device IP Configuration Update")
    print("=" * 70)
    print()
    print(f"📍 New IP Address: {new_ip}")
    print()
    
    # Get base directory
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent  # Go up to Dexmate root
    
    print(f"📁 Base Directory: {base_dir}")
    print()
    
    # Confirm
    response = input("Continue with update? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled")
        sys.exit(0)
    
    print()
    
    # Update files
    updated, failed = update_files(new_ip, base_dir)
    
    # Summary
    print()
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✅ Updated: {len(updated)} files")
    if updated:
        for f in updated:
            print(f"   • {f}")
    
    if failed:
        print(f"❌ Failed: {len(failed)} files")
        for f in failed:
            print(f"   • {f}")
    
    print()
    
    if not failed:
        print("🎉 All files updated successfully!")
        print()
        print("You can now test the suction control:")
        print("  cd LGES/suction")
        print("  python3 test_suction.py")
    else:
        print("⚠️  Some files were not updated. Please check manually.")
    
    sys.exit(0 if not failed else 1)

if __name__ == "__main__":
    main()
