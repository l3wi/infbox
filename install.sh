#!/bin/bash
set -uo pipefail

# InfBox Install Script for Qwen3-Coder-480B-A35B-Instruct-FP8
# Single-command installation for multi-GPU inference stack
# Usage: curl -L https://raw.githubusercontent.com/l3wi/infbox/main/install.sh | bash

# Ensure clean exit on errors
trap 'echo "Installation interrupted. Please check the errors above and try again."' ERR

# ================================
# Configuration
# ================================
REPO_URL="https://github.com/l3wi/infbox.git"
INSTALL_DIR="$HOME/infbox"
MODELS_DIR="$HOME/models"
MODEL_NAME="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
MODEL_SAFE_NAME="Qwen_Qwen3-Coder-480B-A35B-Instruct-FP8"
INSTRUCTIONS_FILE="$HOME/INFBOX_README.txt"

# Docker command prefix (set if we need sudo)
DOCKER_SUDO=""

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
info() { echo -e "${BLUE}[INFO]${NC} $*"; }

# ================================
# System Requirements Check
# ================================

check_nvidia() {
    log "Checking NVIDIA drivers..."
    if ! command -v nvidia-smi &> /dev/null; then
        error "NVIDIA drivers not found. Please install NVIDIA drivers first."
        error "Visit: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html"
        error ""
        error "For Ubuntu, you can install drivers with:"
        error "  sudo apt update"
        error "  sudo apt install nvidia-driver-535"
        error "  sudo reboot"
        return 1
    fi
    
    # Get GPU info
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    TOTAL_VRAM=$((GPU_MEMORY * GPU_COUNT / 1000))  # Total VRAM in GB
    
    log "Detected $GPU_COUNT × $GPU_NAME with ${GPU_MEMORY}MB memory each"
    log "Total VRAM: ${TOTAL_VRAM}GB"
    
    # Check if system can run the 480B model
    if [ "$TOTAL_VRAM" -lt 140 ]; then
        error "Insufficient VRAM for Qwen3-Coder-480B model"
        error "Model requires at least 140GB VRAM, but system has ${TOTAL_VRAM}GB"
        error "Consider using a smaller model or adding more GPUs"
        return 1
    fi
    
    return 0
}

install_docker() {
    log "Installing Docker..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    sudo usermod -aG docker $USER
    rm /tmp/get-docker.sh
    
    # Start Docker service
    sudo systemctl start docker
    sudo systemctl enable docker
    
    log "Docker installed successfully"
    warn "Note: You may need to log out and back in for group changes to take effect"
    
    # For the current session, use sudo for docker commands
    DOCKER_SUDO="sudo"
}

install_docker_compose() {
    log "Installing Docker Compose v2..."
    sudo apt-get update -qq
    sudo apt-get install -y docker-compose-plugin
    log "Docker Compose v2 installed successfully"
}

install_nvidia_container_toolkit() {
    log "Installing NVIDIA Container Toolkit..."
    
    # Get distribution info
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    
    # Add NVIDIA Container Toolkit repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    
    # Install the toolkit
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    log "NVIDIA Container Toolkit installed successfully"
}

check_docker() {
    log "Checking Docker installation..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log "Docker not found. Installing Docker..."
        install_docker
    else
        log "Docker is already installed"
        
        # Check if user needs sudo for docker
        if ! docker ps &> /dev/null; then
            if sudo docker ps &> /dev/null; then
                log "Docker requires sudo (user not in docker group)"
                DOCKER_SUDO="sudo"
            else
                error "Docker is not running or not accessible"
                return 1
            fi
        fi
    fi
    
    # Check Docker Compose v2
    if ! ${DOCKER_SUDO:-} docker compose version &> /dev/null 2>&1; then
        log "Docker Compose v2 not found. Installing..."
        install_docker_compose
    else
        log "Docker Compose v2 is already installed"
    fi
    
    # Check NVIDIA Container Toolkit
    log "Checking NVIDIA Container Toolkit..."
    if ! ${DOCKER_SUDO:-} docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null 2>&1; then
        log "NVIDIA Container Toolkit not properly configured. Installing..."
        install_nvidia_container_toolkit
        
        # Give Docker time to restart
        log "Waiting for Docker to restart..."
        sleep 5
        
        # Test again after installation
        log "Testing NVIDIA Container Toolkit..."
        if ! ${DOCKER_SUDO:-} docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null 2>&1; then
            warn "NVIDIA Container Toolkit test failed. This might be resolved by a reboot."
            warn "Continuing with installation anyway..."
            warn "If vLLM fails to start, please reboot and run: cd $INSTALL_DIR && docker compose up -d"
        else
            log "NVIDIA Container Toolkit is working correctly"
        fi
    else
        log "NVIDIA Container Toolkit is properly configured"
    fi
    
    return 0
}

check_system_requirements() {
    log "Checking system requirements..."
    
    # Check sudo access
    if ! sudo -n true 2>/dev/null; then
        sudo -v || {
            error "This script requires sudo access for installing dependencies"
            return 1
        }
    fi
    
    # Check OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        log "Operating System: $PRETTY_NAME"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 500 ]; then
        warn "Low disk space: ${AVAILABLE_SPACE}GB available"
        warn "Model download requires ~200GB, recommend at least 500GB free"
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python3 is not installed. Please install Python 3.8 or later."
        return 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        error "Git is not installed. Please install git."
        return 1
    fi
    
    return 0
}

# ================================
# Installation Functions
# ================================

clone_repository() {
    log "Preparing InfBox repository..."
    
    if [ -d "$INSTALL_DIR" ]; then
        warn "Installation directory already exists: $INSTALL_DIR"
        log "Using existing installation directory"
        cd "$INSTALL_DIR"
        
        # Update to latest version
        if [ -d ".git" ]; then
            log "Updating to latest version..."
            git pull origin main || warn "Could not update repository"
        fi
    else
        log "Cloning InfBox repository..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
}

download_model() {
    log "Preparing to download model: $MODEL_NAME"
    
    mkdir -p "$MODELS_DIR"
    MODEL_PATH="$MODELS_DIR/$MODEL_SAFE_NAME"
    
    # Check if model already exists
    if [ -d "$MODEL_PATH" ] && [ -n "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
        size=$(du -sh "$MODEL_PATH" | cut -f1)
        log "Model already exists at $MODEL_PATH (size: $size)"
        read -p "Skip model download? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            return 0
        fi
        rm -rf "$MODEL_PATH"
    fi
    
    # Ensure pip is installed
    if ! python3 -m pip --version &>/dev/null 2>&1; then
        log "Installing pip..."
        sudo apt-get update -qq
        sudo apt-get install -y python3-pip
    fi
    
    # Install huggingface-hub if needed
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        log "Installing huggingface-hub..."
        python3 -m pip install --user huggingface-hub tqdm
    fi
    
    log "Downloading model to: $MODEL_PATH"
    log "This will take 30-90 minutes depending on your connection..."
    info "Model size: ~200GB"
    
    # Python download script
    python3 << EOF
import os
import sys
from huggingface_hub import snapshot_download

model_name = "$MODEL_NAME"
local_dir = "$MODEL_PATH"
token = os.environ.get('HF_TOKEN', None)

print(f"Starting download...")

try:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
        max_workers=4
    )
    print("\n✓ Model downloaded successfully!")
except KeyboardInterrupt:
    print("\n✗ Download interrupted. You can resume by running the install script again.")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    if "401" in str(e) or "403" in str(e):
        print("\nThis model may require authentication.")
        print("Please set your Hugging Face token:")
        print("  export HF_TOKEN='your_token_here'")
        print("Then run the install script again.")
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        error "Model download failed"
        exit 1
    fi
    
    log "Model downloaded: $(du -sh "$MODEL_PATH" | cut -f1)"
}

configure_infbox() {
    log "Configuring InfBox for Qwen3-Coder-480B..."
    
    # Create config directory
    mkdir -p "$INSTALL_DIR/config"
    
    # Create Caddyfile
    cat > "$INSTALL_DIR/config/Caddyfile" << 'CADDYEOF'
# Caddyfile for Multi-GPU LLM Inference Stack
# Provides automatic HTTPS and reverse proxy

# Listen on port 5555 and expose to internet
:5555 {
    # Reverse proxy ALL requests to vLLM
    reverse_proxy * vllm:8000
    
    # Request/response logging
    log {
        output stdout
        format console
    }
}

# Also keep port 80 for basic HTTP
:80 {
    # Redirect to port 5555
    redir http://{host}:5555{uri} permanent
}
CADDYEOF
    
    # Create .env file
    cat > "$INSTALL_DIR/.env" << EOF
# InfBox configuration for Qwen3-Coder-480B-A35B-Instruct-FP8

# Model configuration
MODEL_NAME=$MODEL_NAME
MODEL_PATH=$MODELS_DIR/$MODEL_SAFE_NAME
PREC=bfloat16
GPU_COUNT=$GPU_COUNT
GPU_UTIL=0.97
CUDA_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))

# Memory configuration
CPU_GB=0
DISK_GB=100

# vLLM configuration
MAX_MODEL_LEN=238000
DTYPE=bfloat16
KV_CACHE_DTYPE=fp8
QUANTIZATION=fp8
VLLM_USE_V1=1
TORCH_CUDA_ARCH_LIST=9.0

# Service ports
VLLM_PORT=8000

# Workspace
WORKSPACE_DIR=$HOME
WATCH_INTERVAL=1
EXTRA_IGNORE_DIRS=infbox,models

# Model storage
MODELS_PATH=$MODELS_DIR

# Logging
LOG_LEVEL=INFO
EOF

    # Update docker-compose.yml to use the FP8 model
    cat > "$INSTALL_DIR/docker-compose.yml" << 'EOF'
services:
  vllm:
    image: vllm/vllm-openai:latest
    command: >
      ${MODEL_PATH}
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size ${GPU_COUNT}
      --dtype ${DTYPE}
      --max-model-len ${MAX_MODEL_LEN}
      --gpu-memory-utilization ${GPU_UTIL}
      --quantization ${QUANTIZATION}
      --kv-cache-dtype ${KV_CACHE_DTYPE}
      --cpu-offload-gb ${CPU_GB}
      --enable-prefix-caching
      --trust-remote-code
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ${MODELS_PATH}:/models:ro
      - ./cache:/cache
      - ${WORKSPACE_DIR}:${WORKSPACE_DIR}:ro
    ports:
      - "${VLLM_PORT}:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
      - HF_HOME=/models
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - VLLM_USE_V1=${VLLM_USE_V1}
      - TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
    networks:
      - llm-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  watcher:
    build:
      context: .
      dockerfile: docker/Dockerfile.watcher
    volumes:
      - ${WORKSPACE_DIR}:${WORKSPACE_DIR}:ro
      - ./scripts:/scripts:ro
      - ./config/watcher-ignore:/etc/watcher-ignore:ro
    environment:
      - WATCH_DIR=${WORKSPACE_DIR}
      - IGNORE_FILE=/etc/watcher-ignore
      - VLLM_ENDPOINT=http://vllm:8000
      - WATCH_INTERVAL=${WATCH_INTERVAL}
      - LOG_LEVEL=${LOG_LEVEL}
      - EXTRA_IGNORE_DIRS=${EXTRA_IGNORE_DIRS}
    networks:
      - llm-network
    restart: unless-stopped
    depends_on:
      - vllm

  caddy:
    image: caddy:2-alpine
    ports:
      - "80:80"
      - "443:443"
      - "443:443/udp"
      - "5555:5555"
    volumes:
      - ./config/Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
      - caddy_config:/config
    networks:
      - llm-network
    restart: unless-stopped

networks:
  llm-network:
    driver: bridge

volumes:
  caddy_data:
  caddy_config:
EOF
}

create_instructions() {
    log "Creating user instructions..."
    
    cat > "$INSTRUCTIONS_FILE" << EOF
========================================
InfBox Installation Complete!
========================================

Model: Qwen3-Coder-480B-A35B-Instruct-FP8
Installation Directory: $INSTALL_DIR
Model Directory: $MODELS_DIR/$MODEL_SAFE_NAME

GPU Configuration:
- GPUs: $GPU_COUNT × $GPU_NAME
- Total VRAM: ${TOTAL_VRAM}GB

Quick Start:
-----------
1. Navigate to installation directory:
   cd $INSTALL_DIR

2. Start the inference server:
   docker compose up -d

3. Monitor logs:
   docker compose logs -f vllm

4. Test the API (internal):
   curl http://localhost:8000/v1/models

5. Test the API (external):
   curl http://YOUR_SERVER_IP:5555/v1/models

6. Send a test request:
   curl http://localhost:5555/v1/chat/completions \\
     -H "Content-Type: application/json" \\
     -d '{
       "model": "Qwen3-Coder-480B-A35B-Instruct-FP8",
       "messages": [{"role": "user", "content": "Hello, can you help me write some code?"}],
       "max_tokens": 100
     }'

Service Management:
------------------
- Stop services: docker compose down
- Restart services: docker compose restart
- View logs: docker compose logs -f
- Update configuration: edit $INSTALL_DIR/.env

API Endpoints:
-------------
Internal (container network):
- Health check: http://localhost:8000/health
- Models list: http://localhost:8000/v1/models
- Chat completions: http://localhost:8000/v1/chat/completions
- Completions: http://localhost:8000/v1/completions

External (internet accessible):
- Health check: http://YOUR_SERVER_IP:5555/health
- Models list: http://YOUR_SERVER_IP:5555/v1/models
- Chat completions: http://YOUR_SERVER_IP:5555/v1/chat/completions
- Completions: http://YOUR_SERVER_IP:5555/v1/completions

Note: Port 5555 is exposed to the internet via Caddy reverse proxy

Workspace Monitoring:
-------------------
The watcher service automatically monitors your home directory ($HOME)
for code changes and maintains context for the LLM.

Troubleshooting:
---------------
- If services fail to start, check GPU availability: nvidia-smi
- For detailed logs: docker compose logs vllm
- Configuration file: $INSTALL_DIR/.env
- Model location: $MODELS_DIR/$MODEL_SAFE_NAME

For more information, visit the repository:
$REPO_URL

========================================
EOF

    log "Instructions saved to: $INSTRUCTIONS_FILE"
    cat "$INSTRUCTIONS_FILE"
}

start_services() {
    log "Starting InfBox services..."
    
    cd "$INSTALL_DIR"
    
    # Stop any existing services
    ${DOCKER_SUDO:-} docker compose down 2>/dev/null || true
    
    # Start services
    if ! ${DOCKER_SUDO:-} docker compose up -d; then
        error "Failed to start services"
        error "Check logs with: cd $INSTALL_DIR && ${DOCKER_SUDO:-} docker compose logs"
        exit 1
    fi
    
    log "Services started. Waiting for vLLM to initialize..."
    log "This may take 5-10 minutes for model loading..."
    
    # Wait for vLLM to be ready
    max_attempts=120  # 10 minutes
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log "✓ vLLM is ready!"
            break
        fi
        sleep 5
        attempt=$((attempt + 1))
        if [ $((attempt % 12)) -eq 0 ]; then
            log "Still waiting for vLLM to initialize... ($((attempt * 5)) seconds)"
        fi
    done
    
    if [ $attempt -eq $max_attempts ]; then
        error "vLLM failed to start within 10 minutes"
        error "Check logs with: cd $INSTALL_DIR && ${DOCKER_SUDO:-} docker compose logs vllm"
        exit 1
    fi
    
    # Test the API
    log "Testing API endpoint..."
    if curl -s http://localhost:8000/v1/models | grep -q "Qwen3-Coder-480B"; then
        log "✓ API is working correctly!"
    else
        warn "API test returned unexpected response"
        warn "Check manually: curl http://localhost:8000/v1/models"
    fi
}

# ================================
# Main Installation Flow
# ================================

main() {
    clear
    cat << 'EOF'
 ___       __  _               
|_ _|_ _  / _|| |__  ___ __ __ 
 | || ' \|  _|| '_ \/ _ \\ \ / 
|___|_||_|_|  |_.__/\___/_\_\ 
                               
Qwen3-Coder-480B Installation Script
EOF
    echo
    
    log "Starting InfBox installation..."
    
    # Check system requirements
    if ! check_system_requirements; then
        error "System requirements check failed. Please resolve the issues above and try again."
        exit 1
    fi
    
    if ! check_nvidia; then
        error "NVIDIA requirements check failed. Please resolve the issues above and try again."
        exit 1
    fi
    
    if ! check_docker; then
        error "Docker requirements check failed. Please resolve the issues above and try again."
        exit 1
    fi
    
    # Clone repository
    clone_repository
    
    # Download model
    download_model
    
    # Configure InfBox
    configure_infbox
    
    # Create instructions
    create_instructions
    
    # Start services
    start_services
    
    echo
    log "==================================="
    log "Installation Complete!"
    log "==================================="
    log ""
    log "InfBox is now running with Qwen3-Coder-480B"
    log "API endpoints:"
    log "  - Internal: http://localhost:8000"
    log "  - External: http://$(hostname -I | awk '{print $1}'):5555"
    log ""
    log "See $INSTRUCTIONS_FILE for detailed instructions"
    log ""
    log "To stop services: cd $INSTALL_DIR && docker compose down"
    log "To view logs: cd $INSTALL_DIR && docker compose logs -f"
    echo
}

# Run main installation
main "$@"