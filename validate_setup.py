#!/usr/bin/env python3
"""
Validation script to check if all required files are present
"""
import os
import sys

required_files = [
    'docker-compose.yml',
    'Dockerfile.orchestrator',
    'Dockerfile.party', 
    'requirements.txt',
    'proto/smpc.proto',
    'orchestrator/main.py',
    'party/main.py',
    'common/utils.py',
    'common/__init__.py'
]

missing_files = []

for file_path in required_files:
    if not os.path.exists(file_path):
        missing_files.append(file_path)

if missing_files:
    print("❌ Missing required files:")
    for file in missing_files:
        print(f"   - {file}")
    print("\nPlease ensure all files are in the correct locations.")
    sys.exit(1)
else:
    print("✅ All required files are present!")
    print("You can now run: docker-compose up --build")

# Check Docker
try:
    import subprocess
    result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Docker is available")
    else:
        print("❌ Docker not found. Please install Docker.")
except FileNotFoundError:
    print("❌ Docker not found. Please install Docker.")

try:
    result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Docker Compose is available")
    else:
        print("❌ Docker Compose not found. Please install Docker Compose.")
except FileNotFoundError:
    print("❌ Docker Compose not found. Please install Docker Compose.")
