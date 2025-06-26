#!/bin/bash

# ===============================================================================
# SMPC-LCA Computation Simulation - Environment Setup Script
# ===============================================================================
# This script creates a virtual environment and installs all required dependencies
# for the enhanced simulation with automatic GPU detection and CPU fallback.
#
# Usage: source setup.sh [environment_name]
#   OR:  . setup.sh [environment_name]
# If no environment name is provided, defaults to "smpc-lca_simulation_env"
#
# Note: Use 'source' to keep the environment activated after setup completes!
# ===============================================================================

# Check if script is being sourced (recommended) or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "\033[1;33m[WARNING]\033[0m This script should be sourced to keep the environment activated:"
    echo -e "  \033[0;32msource setup.sh\033[0m  or  \033[0;32m. setup.sh\033[0m"
    echo ""
    echo "Continuing anyway, but you'll need to manually activate the environment after setup..."
    echo ""
fi

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default environment name
ENV_NAME=${1:-"smpc-lca_simulation_env"}

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}SMPC-LCA Simulation - Environment Setup${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${BLUE}RECOMMENDED USAGE:${NC} ${YELLOW}source setup.sh${NC} ${BLUE}(keeps environment active after setup)${NC}"
echo ""

# Function to print colored status messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect which Python command to use
detect_python_command() {
    # First try python3
    if command -v python3 &> /dev/null; then
        echo "python3"
        return 0
    # If python3 not found, try python
    elif command -v python &> /dev/null; then
        # Check if python points to Python 3.x
        local python_version=$(python --version 2>&1 | grep -oP 'Python \K[0-9]+\.[0-9]+\.[0-9]+')
        local python_major=$(echo $python_version | cut -d'.' -f1)
        
        if [ "$python_major" -eq 3 ]; then
            echo "python"
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}

# Detect Python command
PYTHON_CMD=$(detect_python_command)

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python $PYTHON_VERSION detected, but Python 3.8+ is required"
    print_error "Please upgrade Python and try again"
    exit 1
fi

print_success "Found Python $PYTHON_VERSION using '$PYTHON_CMD' command"

# Check if pip is available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_error "pip is not available"
    print_error "Please install pip and try again"
    exit 1
fi

print_success "pip is available"

# Check if virtual environment already exists
if [ -d "$ENV_NAME" ]; then
    print_warning "Virtual environment '$ENV_NAME' already exists"
    echo -n "Do you want to remove it and create a new one? (y/N): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Removing existing environment..."
        rm -rf "$ENV_NAME"
        print_success "Existing environment removed"
    else
        print_status "Using existing environment"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$ENV_NAME" ]; then
    print_status "Creating virtual environment '$ENV_NAME'..."
    $PYTHON_CMD -m venv "$ENV_NAME"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$ENV_NAME/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip
print_success "pip upgraded"

# Install core dependencies
print_status "Installing core dependencies..."
echo -e "${BLUE}Installing numpy...${NC}"
pip install numpy

echo -e "${BLUE}Installing pandas...${NC}"
pip install pandas

echo -e "${BLUE}Installing scipy...${NC}"
pip install scipy

print_success "Core dependencies installed successfully"

# GPU Detection and CuPy Installation
print_status "Checking for NVIDIA GPU and CUDA support..."

# Function to detect CUDA version
detect_cuda_version() {
    local cuda_version=""
    
    # Try nvcc first
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        if [ -n "$cuda_version" ]; then
            echo "$cuda_version"
            return 0
        fi
    fi
    
    # Try nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p')
        if [ -n "$cuda_version" ]; then
            echo "$cuda_version"
            return 0
        fi
    fi
    
    return 1
}

# Function to get appropriate CuPy package
get_cupy_package() {
    local cuda_version="$1"
    local major_version=$(echo "$cuda_version" | cut -d'.' -f1)
    local minor_version=$(echo "$cuda_version" | cut -d'.' -f2)
    
    # Convert to comparable number
    local version_num=$((major_version * 100 + minor_version))
    
    if [ "$version_num" -ge 1200 ]; then
        echo "cupy-cuda12x"
    elif [ "$version_num" -ge 1100 ]; then
        echo "cupy-cuda11x"
    else
        echo ""
    fi
}

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        
        # Detect CUDA version
        if cuda_version=$(detect_cuda_version); then
            print_success "CUDA version $cuda_version detected"
            
            # Get appropriate CuPy package
            cupy_package=$(get_cupy_package "$cuda_version")
            
            if [ -n "$cupy_package" ]; then
                print_status "Installing $cupy_package for GPU acceleration..."
                if pip install "$cupy_package"; then
                    print_success "CuPy installed successfully - GPU acceleration enabled"
                else
                    print_warning "CuPy installation failed - will use CPU mode"
                fi
            else
                print_warning "CUDA version $cuda_version is not supported by CuPy"
                print_warning "Will use CPU mode only"
            fi
        else
            print_warning "Could not detect CUDA version"
            print_warning "Will use CPU mode only"
        fi
    else
        print_warning "nvidia-smi command failed"
        print_warning "Will use CPU mode only"
    fi
else
    print_warning "No NVIDIA GPU detected (nvidia-smi not found)"
    print_warning "Will use CPU mode only"
fi

# Optional dependencies for enhanced functionality
print_status "Installing optional dependencies..."

echo -e "${BLUE}Installing psutil for system monitoring...${NC}"
if pip install psutil; then
    print_success "psutil installed"
else
    print_warning "psutil installation failed (optional)"
fi

echo -e "${BLUE}Installing matplotlib for plotting (optional)...${NC}"
if pip install matplotlib; then
    print_success "matplotlib installed"
else
    print_warning "matplotlib installation failed (optional)"
fi

# Create activation script
print_status "Creating activation script..."
cat > activate_env.sh << EOF
#!/bin/bash
# Activation script for MPC simulation environment
echo "Activating MPC simulation environment..."
source "$ENV_NAME/bin/activate"
echo "Environment activated. You can now run the simulation script."
echo "Run: $PYTHON_CMD main.py"
echo "To deactivate, run: deactivate"
EOF

chmod +x activate_env.sh
print_success "Activation script 'activate_env.sh' created"

# Create requirements.txt for future reference
print_status "Creating requirements.txt..."
pip freeze > requirements.txt
print_success "requirements.txt created"

# Final summary
echo ""
echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}SETUP COMPLETE${NC}"
echo -e "${GREEN}===============================================================================${NC}"
print_success "Virtual environment '$ENV_NAME' is ready"

# Check what's installed
echo ""
print_status "Installed packages summary:"
echo -e "  ${BLUE}Core dependencies:${NC}"
echo -e "    - numpy: $($PYTHON_CMD -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "Not found")"
echo -e "    - pandas: $($PYTHON_CMD -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "Not found")"
echo -e "    - scipy: $($PYTHON_CMD -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "Not found")"

echo -e "  ${BLUE}GPU acceleration:${NC}"
if $PYTHON_CMD -c "import cupy" 2>/dev/null; then
    cupy_version=$($PYTHON_CMD -c "import cupy; print(cupy.__version__)" 2>/dev/null)
    print_success "    - cupy: $cupy_version (GPU mode available)"
else
    print_warning "    - cupy: Not installed (CPU mode only)"
fi

echo -e "  ${BLUE}Optional packages:${NC}"
echo -e "    - psutil: $($PYTHON_CMD -c "import psutil; print(psutil.__version__)" 2>/dev/null || echo "Not installed")"
echo -e "    - matplotlib: $($PYTHON_CMD -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null || echo "Not installed")"

echo ""
print_status "System Information:"
echo -e "  ${BLUE}Python Command:${NC} $PYTHON_CMD"
echo -e "  ${BLUE}Python Version:${NC} $PYTHON_VERSION"

echo ""
# Check if environment is currently activated
if [[ "$VIRTUAL_ENV" == *"$ENV_NAME"* ]]; then
    echo -e "${GREEN}Environment is currently ACTIVE and ready to use!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Run the simulation: ${YELLOW}$PYTHON_CMD main.py${NC}"
    echo -e "  2. When done, deactivate: ${YELLOW}deactivate${NC}"
    echo ""
    echo -e "${BLUE}For future sessions:${NC}"
    echo -e "  - Activate manually: ${YELLOW}source $ENV_NAME/bin/activate${NC}"
    echo -e "  - OR use the activation script: ${YELLOW}./activate_env.sh${NC}"
else
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Activate the environment: ${YELLOW}source $ENV_NAME/bin/activate${NC}"
    echo -e "     OR use the activation script: ${YELLOW}./activate_env.sh${NC}"
    echo -e "  2. Run the simulation: ${YELLOW}$PYTHON_CMD main.py${NC}"
    echo -e "  3. When done, deactivate: ${YELLOW}deactivate${NC}"
fi

echo ""
echo -e "${BLUE}Files created:${NC}"
echo -e "  - Virtual environment: ${YELLOW}$ENV_NAME/${NC}"
echo -e "  - Activation script: ${YELLOW}activate_env.sh${NC}"
echo -e "  - Requirements file: ${YELLOW}requirements.txt${NC}"

echo ""
print_success "Setup completed successfully!"
echo ""
if [[ "$VIRTUAL_ENV" == *"$ENV_NAME"* ]]; then
    echo -e "${GREEN} Your environment is ACTIVE and ready for simulation!${NC}"
    echo -e "Run: ${YELLOW}$PYTHON_CMD main.py${NC}"
else
    echo -e "${YELLOW} TIP: Next time use 'source setup.sh' to auto-activate the environment!${NC}"
fi