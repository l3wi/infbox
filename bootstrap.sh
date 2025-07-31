#!/bin/bash
set -euo pipefail

# Multi-GPU Code-Aware LLM Inference Stack Bootstrap Script
# Supports Ubuntu 22.04 with NVIDIA GPUs

echo "=== Multi-GPU LLM Inference Stack Bootstrap ==="
echo "Starting setup at $(date)"

# Detect GPU and set appropriate profile
detect_gpu_profile() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
    
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    
    echo "Detected $GPU_COUNT GPU(s) with ${GPU_MEMORY}MB memory each"
    
    # A6000 has 48GB, use dev profile
    if [ "$GPU_MEMORY" -ge 40000 ] && [ "$GPU_COUNT" -eq 1 ]; then
        echo "Using dev32 profile (single high-memory GPU)"
        cp .env.dev32 .env
    elif [ "$GPU_COUNT" -ge 4 ]; then
        echo "Using prod480 profile (multi-GPU setup)"
        cp .env.prod480 .env
    else
        echo "Using dev32 profile (default)"
        cp .env.dev32 .env
    fi
}

# Check Python installation
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Python3 not found. Installing..."
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-pip
        echo "Python3 and pip3 installed"
    else
        echo "Python3 is already installed"
        # Check pip separately
        if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
            echo "pip3 not found. Installing..."
            sudo apt-get update -qq
            sudo apt-get install -y python3-pip
            echo "pip3 installed"
        fi
    fi
}

# Check Docker installation
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Docker not found. Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        echo "Docker installed. Please log out and back in for group changes to take effect."
    else
        echo "Docker is already installed"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "Docker Compose not found. Installing..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    else
        echo "Docker Compose is already installed"
    fi
}

# Check NVIDIA Container Toolkit
check_nvidia_toolkit() {
    if ! docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
        echo "NVIDIA Container Toolkit not found. Installing..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
    else
        echo "NVIDIA Container Toolkit is already installed"
    fi
}

# Create necessary directories
setup_directories() {
    echo "Setting up directories..."
    mkdir -p models cache workspace logs
    chmod 755 models cache workspace logs
}

# Validate environment files exist
validate_env_files() {
    if [ ! -f ".env.dev32" ]; then
        echo "ERROR: .env.dev32 not found. Please ensure all files are present."
        exit 1
    fi
    if [ ! -f ".env.prod480" ]; then
        echo "ERROR: .env.prod480 not found. Please ensure all files are present."
        exit 1
    fi
}

# Main execution
main() {
    echo "Checking system requirements..."
    check_python
    check_docker
    check_nvidia_toolkit
    
    echo "Validating configuration files..."
    validate_env_files
    
    echo "Detecting GPU configuration..."
    detect_gpu_profile
    
    echo "Setting up directories..."
    setup_directories
    
    echo ""
    echo "=== Bootstrap Complete ==="
    echo "Next steps:"
    echo "1. Review the selected profile in .env"
    echo "2. Download the model (65GB): make fetch-models"
    echo "3. Run 'make start' to launch the stack"
    echo "4. Monitor logs with 'docker-compose logs -f'"
    echo ""
    echo "The system will be available at:"
    echo "  - vLLM API: http://localhost:8000"
    echo "  - OpenAI-compatible endpoint: http://localhost:8000/v1"
    echo ""
    echo "Workspace monitoring:"
    echo "  - Watching: $HOME (your home directory)"
    echo "  - To change: edit WORKSPACE_HOST in .env"
    echo "  - Examples: WORKSPACE_HOST=~/projects or WORKSPACE_HOST=/path/to/code"
}

# Run main function
main "$@"