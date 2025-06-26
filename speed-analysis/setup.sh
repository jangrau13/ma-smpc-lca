#!/bin/bash

# ===============================================================================
# Comprehensive Matrix Operations & Network Performance Benchmark - Environment Setup Script
# ===============================================================================
# This script creates a virtual environment and installs all required dependencies
# for the matrix operations and network performance benchmark suite with automatic
# GPU detection and CPU fallback support.
#
# Usage: source setup.sh [environment_name]
#   OR:  . setup.sh [environment_name]
# If no environment name is provided, defaults to "matrix_benchmark_env"
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
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default environment name
ENV_NAME=${1:-"matrix_benchmark_env"}

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Comprehensive Matrix Operations & Network Performance Benchmark - Environment Setup${NC}"
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

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
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

print_success "Found Python $PYTHON_VERSION using '$PYTHON_CMD' command ✓"

# Check if pip is available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_error "pip is not available"
    print_error "Please install pip and try again"
    exit 1
fi

print_success "pip is available ✓"

# GPU Detection Function
detect_gpu_and_cuda() {
    local gpu_vendor="unknown"
    local cuda_version=""
    
    print_step "Detecting GPU and CUDA capabilities..."
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi > /dev/null 2>&1; then
            gpu_vendor="nvidia"
            print_success "NVIDIA GPU detected ✓"
            
            # Try to get CUDA version from nvcc
            if command -v nvcc &> /dev/null; then
                cuda_version=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
                if [ -n "$cuda_version" ]; then
                    print_success "CUDA $cuda_version detected via nvcc ✓"
                fi
            fi
            
            # Fallback to nvidia-smi for CUDA version
            if [ -z "$cuda_version" ]; then
                cuda_version=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
                if [ -n "$cuda_version" ]; then
                    print_success "CUDA $cuda_version detected via nvidia-smi ✓"
                fi
            fi
            
            if [ -z "$cuda_version" ]; then
                print_warning "NVIDIA GPU found but CUDA version could not be determined"
                print_warning "CuPy installation may fail - will fallback to CPU mode"
            fi
        else
            print_warning "nvidia-smi found but GPU not accessible"
        fi
    else
        print_status "No NVIDIA GPU detected - will use CPU mode"
    fi
    
    echo "$gpu_vendor|$cuda_version"
}

# Get CuPy package name based on CUDA version
get_cupy_package() {
    local cuda_version="$1"
    
    if [ -z "$cuda_version" ]; then
        echo ""
        return
    fi
    
    # Convert version to comparable format
    local major=$(echo "$cuda_version" | cut -d'.' -f1)
    local minor=$(echo "$cuda_version" | cut -d'.' -f2)
    local version_num=$((major * 100 + minor))
    
    if [ $version_num -ge 1200 ]; then
        echo "cupy-cuda12x"
    elif [ $version_num -ge 1100 ]; then
        echo "cupy-cuda11x"
    elif [ $version_num -ge 1000 ]; then
        echo "cupy-cuda110"
    else
        echo ""
    fi
}

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
    print_step "Creating virtual environment '$ENV_NAME'..."
    $PYTHON_CMD -m venv "$ENV_NAME"
    print_success "Virtual environment created ✓"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source "$ENV_NAME/bin/activate"
print_success "Virtual environment activated ✓"

# Upgrade pip
print_step "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip --quiet
print_success "pip upgraded ✓"

# Install core dependencies
print_step "Installing core dependencies..."

echo -e "${CYAN}Installing numpy (required for all matrix operations)...${NC}"
pip install numpy --quiet
NUMPY_VERSION=$($PYTHON_CMD -c "import numpy; print(numpy.__version__)" 2>/dev/null)
print_success "numpy $NUMPY_VERSION installed ✓"

# GPU detection and CuPy installation
GPU_INFO=$(detect_gpu_and_cuda)
GPU_VENDOR=$(echo "$GPU_INFO" | cut -d'|' -f1)
CUDA_VERSION=$(echo "$GPU_INFO" | cut -d'|' -f2)

CUPY_INSTALLED=false
if [ "$GPU_VENDOR" = "nvidia" ] && [ -n "$CUDA_VERSION" ]; then
    CUPY_PACKAGE=$(get_cupy_package "$CUDA_VERSION")
    
    if [ -n "$CUPY_PACKAGE" ]; then
        print_step "Installing CuPy for GPU acceleration..."
        echo -e "${CYAN}Installing $CUPY_PACKAGE (CUDA $CUDA_VERSION support)...${NC}"
        
        if pip install "$CUPY_PACKAGE" --quiet; then
            CUPY_VERSION=$($PYTHON_CMD -c "import cupy; print(cupy.__version__)" 2>/dev/null)
            print_success "$CUPY_PACKAGE $CUPY_VERSION installed ✓"
            print_success "GPU acceleration enabled via CuPy ✓"
            CUPY_INSTALLED=true
        else
            print_warning "$CUPY_PACKAGE installation failed"
            print_warning "Falling back to CPU-only mode"
        fi
    else
        print_warning "CUDA $CUDA_VERSION is not supported by available CuPy packages"
        print_warning "Falling back to CPU-only mode"
    fi
else
    if [ "$GPU_VENDOR" = "nvidia" ]; then
        print_warning "NVIDIA GPU detected but CUDA version unknown"
        print_warning "Skipping CuPy installation - will use CPU mode"
    else
        print_status "No compatible GPU detected - using CPU mode"
    fi
fi

# Install optional performance packages
print_step "Installing optional performance and utility packages..."

echo -e "${CYAN}Installing psutil (system monitoring and memory management)...${NC}"
if pip install psutil --quiet; then
    PSUTIL_VERSION=$($PYTHON_CMD -c "import psutil; print(psutil.__version__)" 2>/dev/null)
    print_success "psutil $PSUTIL_VERSION installed ✓"
else
    print_warning "psutil installation failed (optional but recommended)"
fi

# Create activation script
print_step "Creating activation script..."
cat > activate_benchmark_env.sh << EOF
#!/bin/bash
# Activation script for Matrix Benchmark environment
echo -e "\033[0;34mActivating Matrix Benchmark environment...\033[0m"
source "$ENV_NAME/bin/activate"
echo -e "\033[0;32mEnvironment activated. You can now run the benchmark suite.\033[0m"
echo ""
echo -e "\033[0;33mAvailable commands:\033[0m"
echo -e "  \033[0;36mTest environment:\033[0m $PYTHON_CMD test_benchmark_environment.py"
echo -e "  \033[0;36mRun benchmark:\033[0m $PYTHON_CMD matrix_benchmark.py"
echo -e "  \033[0;36mDeactivate:\033[0m deactivate"
EOF

chmod +x activate_benchmark_env.sh
print_success "Activation script 'activate_benchmark_env.sh' created ✓"

# Create requirements.txt for future reference
print_step "Creating requirements.txt..."
pip freeze > requirements.txt
print_success "requirements.txt created ✓"

# Final summary
echo ""
echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}SETUP COMPLETE${NC}"
echo -e "${GREEN}===============================================================================${NC}"
print_success "Virtual environment '$ENV_NAME' is ready for matrix benchmarking!"

# Check what's installed
echo ""
print_status "Installed packages summary:"
echo -e "  ${BLUE}Core Dependencies:${NC}"
echo -e "    - numpy: $NUMPY_VERSION (CPU matrix operations)"

if [ "$CUPY_INSTALLED" = true ]; then
    CUPY_VERSION=$($PYTHON_CMD -c "import cupy; print(cupy.__version__)" 2>/dev/null)
    echo -e "    - cupy: $CUPY_VERSION (GPU acceleration)"
    COMPUTATION_MODE="GPU + CPU"
else
    echo -e "    - cupy: Not installed (CPU mode only)"
    COMPUTATION_MODE="CPU only"
fi

echo -e "  ${BLUE}Optional Packages:${NC}"
echo -e "    - psutil: $($PYTHON_CMD -c "import psutil; print(psutil.__version__)" 2>/dev/null || echo "Not installed")"

echo ""
print_status "System Capabilities:"
echo -e "  ${BLUE}Python Command:${NC} $PYTHON_CMD"
echo -e "  ${BLUE}Python Version:${NC} $PYTHON_VERSION"
echo -e "  ${BLUE}Computation Mode:${NC} $COMPUTATION_MODE"
if [ "$GPU_VENDOR" = "nvidia" ]; then
    echo -e "  ${BLUE}GPU Vendor:${NC} NVIDIA"
    if [ -n "$CUDA_VERSION" ]; then
        echo -e "  ${BLUE}CUDA Version:${NC} $CUDA_VERSION"
    fi
fi

echo ""
# Check if environment is currently activated
if [[ "$VIRTUAL_ENV" == *"$ENV_NAME"* ]]; then
    echo -e "${GREEN}🎉 Environment is currently ACTIVE and ready to use!${NC}"
    echo ""
    echo -e "${BLUE}For future sessions:${NC}"
    echo -e "  - Activate manually: ${YELLOW}source $ENV_NAME/bin/activate${NC}"
    echo -e "  - OR use the activation script: ${YELLOW}./activate_benchmark_env.sh${NC}"
else
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Activate the environment: ${YELLOW}source $ENV_NAME/bin/activate${NC}"
    echo -e "     OR use the activation script: ${YELLOW}./activate_benchmark_env.sh${NC}"
    echo -e "  3. When done, deactivate: ${YELLOW}deactivate${NC}"
fi

echo ""
echo -e "${BLUE}Files created:${NC}"
echo -e "  - Virtual environment: ${YELLOW}$ENV_NAME/${NC}"
echo -e "  - Activation script: ${YELLOW}activate_benchmark_env.sh${NC}"
echo -e "  - Requirements file: ${YELLOW}requirements.txt${NC}"


echo ""
print_success "Setup completed successfully!"
echo ""
if [[ "$VIRTUAL_ENV" == *"$ENV_NAME"* ]]; then
    echo -e "${GREEN}Your environment is ACTIVE and ready for matrix benchmarking!${NC}"
    echo -e "Run: ${YELLOW}$PYTHON_CMD test_benchmark_environment.py${NC} to verify GPU/CPU capabilities"
else
    echo -e "${YELLOW}TIP: Next time use 'source setup.sh' to auto-activate the environment!${NC}"
fi
