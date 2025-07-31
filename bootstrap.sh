#!/bin/bash
set -euo pipefail

# Multi-GPU Code-Aware LLM Inference Stack Bootstrap Script
# One-command setup: environment, models, and server launch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# =============================================================================
# SYSTEM REQUIREMENTS CHECK
# =============================================================================

check_ubuntu() {
    if [ ! -f /etc/os-release ]; then
        error "Cannot detect OS. This script requires Ubuntu 22.04"
        exit 1
    fi
    
    . /etc/os-release
    if [[ "$ID" != "ubuntu" ]] || [[ ! "$VERSION_ID" =~ ^22\. ]]; then
        error "This script requires Ubuntu 22.04. Detected: $ID $VERSION_ID"
        exit 1
    fi
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log "Installing Python3..."
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-pip python3-venv
    fi
    
    # Ensure pip is available
    if ! python3 -m pip --version &> /dev/null; then
        log "Installing pip..."
        sudo apt-get install -y python3-pip
    fi
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        warn "Docker installed. You may need to log out and back in for group changes."
    fi
    
    # Install Docker Compose v2
    if ! docker compose version &> /dev/null; then
        log "Installing Docker Compose v2..."
        sudo apt-get update -qq
        sudo apt-get install -y docker-compose-plugin
    fi
}

check_nvidia() {
    if ! command -v nvidia-smi &> /dev/null; then
        error "NVIDIA drivers not found. Please install NVIDIA drivers first."
        error "Visit: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html"
        exit 1
    fi
    
    # Check NVIDIA Container Toolkit
    if ! docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null 2>&1; then
        log "Installing NVIDIA Container Toolkit..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        sudo apt-get update -qq
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    fi
}

# =============================================================================
# GPU DETECTION AND PROFILE SELECTION
# =============================================================================

detect_gpu_profile() {
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    
    log "Detected $GPU_COUNT × $GPU_NAME with ${GPU_MEMORY}MB memory"
    
    # Auto-select profile based on GPU
    if [ "$GPU_MEMORY" -ge 40000 ] && [ "$GPU_COUNT" -eq 1 ]; then
        PROFILE="dev32"
        log "Selected profile: dev32 (single high-memory GPU)"
    elif [ "$GPU_COUNT" -ge 4 ]; then
        PROFILE="prod480"
        log "Selected profile: prod480 (multi-GPU setup)"
    else
        PROFILE="dev32"
        log "Selected profile: dev32 (default)"
    fi
}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup_environment() {
    # Create necessary directories
    mkdir -p models cache workspace logs
    
    # Create environment files if they don't exist
    if [ ! -f .env.dev32 ]; then
        cat > .env.dev32 << 'EOF'
# Development configuration for single GPU (A6000/4090)
# Qwen2.5-Coder-32B with INT4 quantization

# Model configuration
MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
PREC=auto
GPU_COUNT=1
GPU_UTIL=0.90
CUDA_DEVICES=0

# Memory configuration
CPU_GB=20
DISK_GB=50

# vLLM configuration
MAX_MODEL_LEN=8192
DTYPE=half
KV_CACHE_DTYPE=fp16

# Service ports
VLLM_PORT=8000
LMCACHE_PORT=8100
CAKE_PORT=8200

# Workspace
WORKSPACE_HOST=~/
WORKSPACE_DIR=/workspace
WATCH_INTERVAL=1

# Model storage
MODELS_PATH=./models

# Logging
LOG_LEVEL=INFO
EOF
    fi
    
    if [ ! -f .env.prod480 ]; then
        cat > .env.prod480 << 'EOF'
# Production configuration for multi-GPU cluster
# Qwen3-Coder-480B with FP8 precision

# Model configuration
MODEL_NAME=Qwen/Qwen3-Coder-480B-Instruct-FP8
PREC=fp8
GPU_COUNT=4
GPU_UTIL=0.85
CUDA_DEVICES=0,1,2,3

# Memory configuration
CPU_GB=40
DISK_GB=100

# vLLM configuration
MAX_MODEL_LEN=131072
DTYPE=float8
KV_CACHE_DTYPE=fp16

# Service ports
VLLM_PORT=8000
LMCACHE_PORT=8100
CAKE_PORT=8200

# Workspace
WORKSPACE_HOST=~/
WORKSPACE_DIR=/workspace
WATCH_INTERVAL=1

# Model storage
MODELS_PATH=./models

# Logging
LOG_LEVEL=INFO
EOF
    fi
    
    # Copy selected profile
    cp .env.$PROFILE .env
    log "Environment configured with $PROFILE profile"
}

# =============================================================================
# MODEL DOWNLOAD
# =============================================================================

download_model() {
    # Load environment
    export $(cat .env | grep -v '^#' | xargs)
    
    MODEL_DIR="$MODELS_PATH/$(echo $MODEL_NAME | tr '/' '_')"
    
    # Check if model exists
    if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
        size=$(du -sh "$MODEL_DIR" | cut -f1)
        log "Model already exists (size: $size)"
        read -p "Re-download? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
        rm -rf "$MODEL_DIR"
    fi
    
    # Install huggingface-hub
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        log "Installing huggingface-hub..."
        pip3 install -U huggingface-hub tqdm
    fi
    
    log "Downloading model: $MODEL_NAME"
    log "This may take 30-60 minutes..."
    
    # Python download script with progress
    python3 << EOF
import os
import sys
from huggingface_hub import snapshot_download

model_name = "$MODEL_NAME"
local_dir = "$MODEL_DIR"
token = os.environ.get('HF_TOKEN', None)

print(f"Downloading to: {local_dir}")

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
    print("\n✗ Download interrupted")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        error "Model download failed"
        exit 1
    fi
    
    log "Model downloaded: $(du -sh "$MODEL_DIR" | cut -f1)"
}

# =============================================================================
# DOCKER COMPOSE SETUP
# =============================================================================

create_docker_compose() {
    # Create docker-compose.yml if it doesn't exist
    if [ ! -f docker-compose.yml ]; then
        cat > docker-compose.yml << 'EOF'
services:
  vllm:
    image: vllm/vllm-openai:latest
    command: >
      ${MODEL_NAME}
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size ${GPU_COUNT:-1}
      --dtype ${DTYPE:-auto}
      --max-model-len ${MAX_MODEL_LEN:-32768}
      --gpu-memory-utilization ${GPU_UTIL:-0.85}
      --enable-prefix-caching
      --enforce-eager
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ${MODELS_PATH:-./models}:/models:ro
      - ./cache:/cache
      - ${WORKSPACE_HOST:-~/}:/workspace:ro
    ports:
      - "${VLLM_PORT:-8000}:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-}
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
      dockerfile: Dockerfile.watcher
    volumes:
      - ${WORKSPACE_HOST:-~/}:/workspace:ro
      - ./scripts:/scripts:ro
    environment:
      - WATCH_DIR=${WORKSPACE_DIR:-/workspace}
      - IGNORE_FILE=.gitignore
      - VLLM_ENDPOINT=http://vllm:8000
      - WATCH_INTERVAL=${WATCH_INTERVAL:-1}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
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
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
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
    fi
    
    # Create docker-compose.override.yml with local model path
    export $(cat .env | grep -v '^#' | xargs)
    MODEL_DIR="$MODELS_PATH/$(echo $MODEL_NAME | tr '/' '_')"
    
    cat > docker-compose.override.yml << EOF
services:
  vllm:
    command: >
      /models/$(basename "$MODEL_DIR")
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size \${GPU_COUNT:-1}
      --dtype \${DTYPE:-auto}
      --max-model-len \${MAX_MODEL_LEN:-8192}
      --gpu-memory-utilization \${GPU_UTIL:-0.85}
      --enable-prefix-caching
      --enforce-eager
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
}

# =============================================================================
# SERVICE MANAGEMENT
# =============================================================================

start_services() {
    log "Starting services..."
    
    # Stop any existing services
    docker compose down 2>/dev/null || true
    
    # Start services
    if ! docker compose up -d; then
        error "Failed to start services"
        exit 1
    fi
    
    log "Services started. Waiting for vLLM to initialize..."
    
    # Wait for vLLM to be ready
    max_attempts=60
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log "vLLM is ready!"
            break
        fi
        sleep 5
        attempt=$((attempt + 1))
        echo -n "."
    done
    echo
    
    if [ $attempt -eq $max_attempts ]; then
        error "vLLM failed to start. Check logs with: docker compose logs vllm"
        exit 1
    fi
}

test_api() {
    log "Testing API..."
    
    # Test models endpoint
    if curl -s http://localhost:8000/v1/models | grep -q "model"; then
        log "✓ Models endpoint working"
    else
        error "Models endpoint not responding"
        return 1
    fi
    
    # Test chat completion
    response=$(curl -s http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'$(docker compose exec vllm curl -s http://localhost:8000/v1/models | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | head -1)'",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 50
        }' 2>/dev/null)
    
    if echo "$response" | grep -q "content"; then
        log "✓ Chat completion working"
        return 0
    else
        error "Chat completion failed"
        return 1
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    cat << 'EOF'
 ___       __  _               
|_ _|_ _  / _|| |__  ___ __ __ 
 | || ' \|  _|| '_ \/ _ \\ \ / 
|___|_||_|_|  |_.__/\___/_\_\ 
                               
Multi-GPU Code-Aware LLM Inference Stack
EOF
    echo

    # System checks
    log "Checking system requirements..."
    check_ubuntu
    check_python
    check_docker
    check_nvidia
    
    # GPU detection
    detect_gpu_profile
    
    # Environment setup
    setup_environment
    
    # Model download
    log "Preparing model..."
    download_model
    
    # Docker setup
    log "Configuring Docker services..."
    create_docker_compose
    
    # Start services
    start_services
    
    # Test API
    if test_api; then
        echo
        log "=== Bootstrap Complete! ==="
        log "Services running at:"
        log "  - vLLM API: http://localhost:8000"
        log "  - OpenAI-compatible: http://localhost:8000/v1"
        log ""
        log "Commands:"
        log "  - View logs: docker compose logs -f"
        log "  - Stop services: docker compose down"
        log "  - Restart: docker compose restart"
        echo
    else
        error "API test failed. Check logs: docker compose logs vllm"
        exit 1
    fi
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi