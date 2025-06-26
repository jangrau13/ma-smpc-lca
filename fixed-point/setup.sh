#!/bin/bash

# SMPC System Setup Script
# This script sets up the complete directory structure and files for the SMPC system

set -e

echo "Setting up SMPC Docker System..."

# Create directory structure
echo "Creating directory structure..."
mkdir -p orchestrator
mkdir -p party
mkdir -p common
mkdir -p proto
mkdir -p results

# Create __init__.py files for Python packages
echo "Creating Python package files..."
touch common/__init__.py
touch orchestrator/__init__.py
touch party/__init__.py

# Create results directory with proper permissions
echo "Setting up results directory..."
chmod 755 results

# Generate protobuf files (will be done in Docker, but can be done locally too)
echo "Protobuf files will be generated during Docker build..."

# Create a simple validation script
cat > validate_setup.py << 'EOF'
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
EOF

chmod +x validate_setup.py

# Create a helper script for common operations
cat > manage.sh << 'EOF'
#!/bin/bash

# SMPC System Management Script

case "$1" in
    "build")
        echo "Building SMPC system..."
        docker-compose build
        ;;
    "start")
        echo "Starting SMPC system..."
        docker-compose up -d
        echo "Web interface available at: http://localhost:8080"
        ;;
    "stop")
        echo "Stopping SMPC system..."
        docker-compose down
        ;;
    "restart")
        echo "Restarting SMPC system..."
        docker-compose down
        docker-compose up -d
        ;;
    "rebuild")
        echo "Rebuilding SMPC system from scratch..."
        echo "Stopping containers..."
        docker-compose down
        echo "Removing existing containers and images..."
        docker-compose down --rmi all --volumes --remove-orphans
        echo "Building fresh images..."
        docker-compose build
        echo "Starting system with new build..."
        docker-compose up -d
        echo "✅ Rebuild complete! Web interface available at: http://localhost:8080"
        ;;
    "logs")
        if [ -z "$2" ]; then
            echo "Showing logs for all parties (orchestrator, party1, party2, party3)..."
            docker-compose logs -f orchestrator party1 party2 party3
        else
            docker-compose logs -f "$2"
        fi
        ;;
    "status")
        echo "SMPC System Status:"
        docker-compose ps
        ;;
    "clean")
        echo "Cleaning up SMPC system..."
        docker-compose down
        docker system prune -f
        echo "Cleaned up containers and images"
        ;;
    "shell")
        if [ -z "$2" ]; then
            echo "Usage: $0 shell <service>"
            echo "Available services: orchestrator, party1, party2, party3"
        else
            docker-compose exec "$2" bash
        fi
        ;;
    "validate")
        python3 validate_setup.py
        ;;
    *)
        echo "SMPC System Management"
        echo "Usage: $0 {build|start|stop|restart|rebuild|logs|status|clean|shell|validate}"
        echo ""
        echo "Commands:"
        echo "  build     - Build the Docker images"
        echo "  start     - Start the system in background"
        echo "  stop      - Stop the system"
        echo "  restart   - Restart the system"
        echo "  rebuild   - Complete rebuild with fresh images and containers"
        echo "  logs      - Show logs for all parties (add service name for specific logs)"
        echo "  status    - Show system status"
        echo "  clean     - Clean up containers and images"
        echo "  shell     - Access container shell (specify service)"
        echo "  validate  - Validate setup and check requirements"
        echo ""
        echo "Examples:"
        echo "  $0 logs orchestrator"
        echo "  $0 shell party1"
        echo "  $0 rebuild"
        ;;
esac
EOF

chmod +x manage.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy all the provided files to their respective directories"
echo "2. Run validation: ./validate_setup.py"
echo "3. Build and start: ./manage.sh build && ./manage.sh start"
echo "4. Open web interface: http://localhost:8080"
echo ""
echo "Management commands available in ./manage.sh"