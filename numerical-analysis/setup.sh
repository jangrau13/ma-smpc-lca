#!/bin/bash

# ===============================================================================
# Multi-CSV Data Analysis - Environment Setup Script
# ===============================================================================
# This script creates a virtual environment and installs all required dependencies
# for the multi-CSV data analysis with feature importance and error comparison.
#
# Usage: source setup.sh [environment_name]
#   OR:  . setup.sh [environment_name]
# If no environment name is provided, defaults to "data_analysis_env"
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
ENV_NAME=${1:-"data_analysis_env"}

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Multi-CSV Data Analysis - Environment Setup${NC}"
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

# Install core data science dependencies
print_step "Installing core data science dependencies..."

echo -e "${CYAN}Installing numpy (numerical computing)...${NC}"
pip install numpy --quiet
NUMPY_VERSION=$($PYTHON_CMD -c "import numpy; print(numpy.__version__)" 2>/dev/null)
print_success "numpy $NUMPY_VERSION installed ✓"

echo -e "${CYAN}Installing pandas (data manipulation)...${NC}"
pip install pandas --quiet
PANDAS_VERSION=$($PYTHON_CMD -c "import pandas; print(pandas.__version__)" 2>/dev/null)
print_success "pandas $PANDAS_VERSION installed ✓"

echo -e "${CYAN}Installing scipy (scientific computing)...${NC}"
pip install scipy --quiet
SCIPY_VERSION=$($PYTHON_CMD -c "import scipy; print(scipy.__version__)" 2>/dev/null)
print_success "scipy $SCIPY_VERSION installed ✓"

# Install machine learning dependencies
print_step "Installing machine learning dependencies..."

echo -e "${CYAN}Installing scikit-learn (machine learning)...${NC}"
pip install scikit-learn --quiet
SKLEARN_VERSION=$($PYTHON_CMD -c "import sklearn; print(sklearn.__version__)" 2>/dev/null)
print_success "scikit-learn $SKLEARN_VERSION installed ✓"

# Install visualization dependencies
print_step "Installing visualization dependencies..."

echo -e "${CYAN}Installing matplotlib (plotting)...${NC}"
pip install matplotlib --quiet
MATPLOTLIB_VERSION=$($PYTHON_CMD -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null)
print_success "matplotlib $MATPLOTLIB_VERSION installed ✓"

echo -e "${CYAN}Installing seaborn (statistical visualization)...${NC}"
pip install seaborn --quiet
SEABORN_VERSION=$($PYTHON_CMD -c "import seaborn; print(seaborn.__version__)" 2>/dev/null)
print_success "seaborn $SEABORN_VERSION installed ✓"

# Optional performance and utility packages
print_step "Installing optional performance and utility packages..."

echo -e "${CYAN}Installing psutil (system monitoring)...${NC}"
if pip install psutil --quiet; then
    PSUTIL_VERSION=$($PYTHON_CMD -c "import psutil; print(psutil.__version__)" 2>/dev/null)
    print_success "psutil $PSUTIL_VERSION installed ✓"
else
    print_warning "psutil installation failed (optional)"
fi


# Create activation script
print_step "Creating activation script..."
cat > activate_env.sh << EOF
#!/bin/bash
# Activation script for Data Analysis environment
echo -e "\033[0;34mActivating Data Analysis environment...\033[0m"
source "$ENV_NAME/bin/activate"
echo -e "\033[0;32mEnvironment activated. You can now run the analysis script.\033[0m"
echo -e "\033[0;33mRun: $PYTHON_CMD analysis.py\033[0m"
echo -e "\033[0;34mTo deactivate, run: deactivate\033[0m"
EOF

chmod +x activate_env.sh
print_success "Activation script 'activate_env.sh' created ✓"

# Create requirements.txt for future reference
print_step "Creating requirements.txt..."
pip freeze > requirements.txt
print_success "requirements.txt created ✓"

# Final summary
echo ""
echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}SETUP COMPLETE${NC}"
echo -e "${GREEN}===============================================================================${NC}"
print_success "Virtual environment '$ENV_NAME' is ready for data analysis!"

# Check what's installed
echo ""
print_status "Installed packages summary:"
echo -e "  ${BLUE}Core Data Science:${NC}"
echo -e "    - numpy: $NUMPY_VERSION"
echo -e "    - pandas: $PANDAS_VERSION" 
echo -e "    - scipy: $SCIPY_VERSION"

echo -e "  ${BLUE}Machine Learning:${NC}"
echo -e "    - scikit-learn: $SKLEARN_VERSION"

echo -e "  ${BLUE}Visualization:${NC}"
echo -e "    - matplotlib: $MATPLOTLIB_VERSION"
echo -e "    - seaborn: $SEABORN_VERSION"

echo -e "  ${BLUE}Optional Packages:${NC}"
echo -e "    - psutil: $($PYTHON_CMD -c "import psutil; print(psutil.__version__)" 2>/dev/null || echo "Not installed")"
echo -e "    - openpyxl: $($PYTHON_CMD -c "import openpyxl; print(openpyxl.__version__)" 2>/dev/null || echo "Not installed")"
echo -e "    - xlrd: $($PYTHON_CMD -c "import xlrd; print(xlrd.__version__)" 2>/dev/null || echo "Not installed")"
echo -e "    - jupyter: $($PYTHON_CMD -c "import jupyter; print('installed')" 2>/dev/null || echo "Not installed")"

echo ""
print_status "System Information:"
echo -e "  ${BLUE}Python Command:${NC} $PYTHON_CMD"
echo -e "  ${BLUE}Python Version:${NC} $PYTHON_VERSION"

echo ""
# Check if environment is currently activated
if [[ "$VIRTUAL_ENV" == *"$ENV_NAME"* ]]; then
    echo -e "${GREEN}🎉 Environment is currently ACTIVE and ready to use!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Test the environment: ${YELLOW}$PYTHON_CMD test_environment.py${NC}"
    echo -e "  2. Run the analysis: ${YELLOW}$PYTHON_CMD analysis.py${NC}"
    echo -e "  3. When done, deactivate: ${YELLOW}deactivate${NC}"
    echo ""
    echo -e "${BLUE}For future sessions:${NC}"
    echo -e "  - Activate manually: ${YELLOW}source $ENV_NAME/bin/activate${NC}"
    echo -e "  - OR use the activation script: ${YELLOW}./activate_env.sh${NC}"
else
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Activate the environment: ${YELLOW}source $ENV_NAME/bin/activate${NC}"
    echo -e "     OR use the activation script: ${YELLOW}./activate_env.sh${NC}"
    echo -e "  2. Run the analysis: ${YELLOW}$PYTHON_CMD analysis.py${NC}"
    echo -e "  3. When done, deactivate: ${YELLOW}deactivate${NC}"
fi

echo ""
echo -e "${BLUE}Files created:${NC}"
echo -e "  - Virtual environment: ${YELLOW}$ENV_NAME/${NC}"
echo -e "  - Activation script: ${YELLOW}activate_env.sh${NC}"
echo -e "  - Requirements file: ${YELLOW}requirements.txt${NC}"
echo -e "  - Sample configuration: ${YELLOW}sample_config.py${NC}"
echo -e "  - Environment test: ${YELLOW}test_environment.py${NC}"

# Test the environment
echo ""
print_step "Testing the environment..."
if $PYTHON_CMD test_environment.py > /dev/null 2>&1; then
    print_success "Environment test passed"
else
    print_warning "Environment test had some issues - run '$PYTHON_CMD test_environment.py' for details"
fi

echo ""
print_success "Setup completed successfully!"
echo ""
if [[ "$VIRTUAL_ENV" == *"$ENV_NAME"* ]]; then
    echo -e "${GREEN}Your environment is ACTIVE and ready for data analysis!${NC}"
    echo -e "Run: ${YELLOW}$PYTHON_CMD test_environment.py${NC} to verify everything works"
else
    echo -e "${YELLOW}TIP: Next time use 'source setup.sh' to auto-activate the environment!${NC}"
fi

echo ""