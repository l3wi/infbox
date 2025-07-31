#!/bin/bash
set -euo pipefail

# InfBox Install Script for Qwen3-Coder-480B-A35B-Instruct-FP8
# Single-command installation for multi-GPU inference stack
# Usage: curl -L https://raw.githubusercontent.com/YOUR_REPO/infbox/main/install.sh | bash

# ================================
# Configuration
# ================================
REPO_URL="https://github.com/l3wi/infbox.git"
INSTALL_DIR="$HOME/infbox"
MODELS_DIR="$HOME/models"
MODEL_NAME="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
MODEL_SAFE_NAME="Qwen_Qwen3-Coder-480B-A35B-Instruct-FP8"
INSTRUCTIONS_FILE="$HOME/INFBOX_README.txt"

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
        exit 1
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
        exit 1
    fi
}

check_docker() {
    log "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed."
        info "Please install Docker first:"
        info "  curl -fsSL https://get.docker.com | sudo sh"
        info "  sudo usermod -aG docker \$USER"
        info "Then log out and back in."
        exit 1
    fi
    
    # Check Docker Compose v2
    if ! docker compose version &> /dev/null 2>&1; then
        error "Docker Compose v2 is not installed."
        info "Please install Docker Compose plugin:"
        info "  sudo apt-get update && sudo apt-get install docker-compose-plugin"
        exit 1
    fi
    
    # Check NVIDIA Container Toolkit
    if ! docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null 2>&1; then
        error "NVIDIA Container Toolkit is not properly configured."
        info "Please install NVIDIA Container Toolkit:"
        info "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        info "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        info "  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        info "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        info "  sudo nvidia-ctk runtime configure --runtime=docker"
        info "  sudo systemctl restart docker"
        exit 1
    fi
}

check_system_requirements() {
    log "Checking system requirements..."
    
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
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        error "Git is not installed. Please install git."
        exit 1
    fi
}

# ================================
# Installation Functions
# ================================

clone_repository() {
    log "Cloning InfBox repository..."
    
    if [ -d "$INSTALL_DIR" ]; then
        warn "Installation directory already exists: $INSTALL_DIR"
        read -p "Remove existing installation and continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Installation cancelled"
            exit 1
        fi
        rm -rf "$INSTALL_DIR"
    fi
    
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
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
    
    # Create .env file
    cat > "$INSTALL_DIR/.env" << EOF
# InfBox configuration for Qwen3-Coder-480B-A35B-Instruct-FP8

# Model configuration
MODEL_NAME=$MODEL_NAME
MODEL_PATH=$MODELS_DIR/$MODEL_SAFE_NAME
PREC=fp8
GPU_COUNT=$GPU_COUNT
GPU_UTIL=0.90
CUDA_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))

# Memory configuration
CPU_GB=40
DISK_GB=100

# vLLM configuration
MAX_MODEL_LEN=32768
DTYPE=float16
KV_CACHE_DTYPE=fp8
QUANTIZATION=fp8

# Service ports
VLLM_PORT=8000

# Workspace
WORKSPACE_DIR=$HOME
WATCH_INTERVAL=1

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
      --enable-prefix-caching
      --trust-remote-code
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
      - ${WORKSPACE_DIR}:/workspace:ro
    ports:
      - "${VLLM_PORT}:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
      - HF_HOME=/models
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
      - ${WORKSPACE_DIR}:/workspace:ro
      - ./scripts:/scripts:ro
    environment:
      - WATCH_DIR=/workspace
      - IGNORE_FILE=.gitignore
      - VLLM_ENDPOINT=http://vllm:8000
      - WATCH_INTERVAL=${WATCH_INTERVAL}
      - LOG_LEVEL=${LOG_LEVEL}
    networks:
      - llm-network
    restart: unless-stopped
    depends_on:
      - vllm

networks:
  llm-network:
    driver: bridge
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

4. Test the API:
   curl http://localhost:8000/v1/models

5. Send a test request:
   curl http://localhost:8000/v1/chat/completions \\
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
- Health check: http://localhost:8000/health
- Models list: http://localhost:8000/v1/models
- Chat completions: http://localhost:8000/v1/chat/completions
- Completions: http://localhost:8000/v1/completions

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
    docker compose down 2>/dev/null || true
    
    # Start services
    if ! docker compose up -d; then
        error "Failed to start services"
        error "Check logs with: cd $INSTALL_DIR && docker compose logs"
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
        error "Check logs with: cd $INSTALL_DIR && docker compose logs vllm"
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
    check_system_requirements
    check_nvidia
    check_docker
    
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
    log "API endpoint: http://localhost:8000"
    log ""
    log "See $INSTRUCTIONS_FILE for detailed instructions"
    log ""
    log "To stop services: cd $INSTALL_DIR && docker compose down"
    log "To view logs: cd $INSTALL_DIR && docker compose logs -f"
    echo
}

# Run main installation
main "$@"