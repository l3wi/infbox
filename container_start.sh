#!/bin/bash
set -euo pipefail

# Multi-GPU Code-Aware LLM Inference Stack - Container Start Script
# This script prepares and launches the entire stack automatically

echo "==========================================="
echo "Multi-GPU LLM Inference Stack Setup"
echo "Starting at $(date)"
echo "==========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to wait for service
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=60
    local attempt=0
    
    echo "Waiting for $service_name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "✓ $service_name is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo "✗ $service_name failed to start after $max_attempts attempts"
    return 1
}

# 1. System Requirements Check
echo ""
echo "=== Checking System Requirements ==="

# Check Python installation
if ! command_exists python3; then
    echo "Python3 not found. Installing..."
    apt-get update -qq
    apt-get install -y python3 python3-pip
    echo "✓ Python3 and pip3 installed"
else
    # Check pip separately
    if ! command_exists pip3 && ! python3 -m pip --version &> /dev/null; then
        echo "pip3 not found. Installing..."
        apt-get update -qq
        apt-get install -y python3-pip
        echo "✓ pip3 installed"
    fi
fi

# Check for NVIDIA GPU
if ! command_exists nvidia-smi; then
    echo "ERROR: NVIDIA GPU not detected. This stack requires CUDA-capable GPUs."
    exit 1
fi

echo "✓ NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  /'

# 2. Install Docker if needed
if ! command_exists docker; then
    echo ""
    echo "=== Installing Docker ==="
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    
    # Add current user to docker group
    usermod -aG docker $USER || true
    
    # Start Docker daemon
    systemctl start docker || service docker start || true
    echo "✓ Docker installed"
else
    echo "✓ Docker already installed"
fi

# 3. Install Docker Compose if needed
if ! command_exists docker-compose; then
    echo ""
    echo "=== Installing Docker Compose ==="
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
    curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "✓ Docker Compose installed"
else
    echo "✓ Docker Compose already installed"
fi

# 4. Install NVIDIA Container Toolkit
echo ""
echo "=== Setting up NVIDIA Container Toolkit ==="
if ! docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - || true
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    
    # Restart Docker to apply changes
    systemctl restart docker || service docker restart || true
    sleep 5
    echo "✓ NVIDIA Container Toolkit installed"
else
    echo "✓ NVIDIA Container Toolkit already configured"
fi

# 5. Clone repository if not already present
if [ ! -f "docker-compose.yml" ]; then
    echo ""
    echo "=== Setting up Repository ==="
    
    # Default to the infbox repository
    GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/l3wi/infbox}"
    
    # Clone into current directory or specified location
    INSTALL_DIR="${INSTALL_DIR:-/opt/infbox}"
    
    if [ ! -d "$INSTALL_DIR" ]; then
        echo "Cloning from $GIT_REPO_URL..."
        git clone "$GIT_REPO_URL" "$INSTALL_DIR"
        echo "✓ Repository cloned to $INSTALL_DIR"
    else
        echo "Repository already exists at $INSTALL_DIR"
        cd "$INSTALL_DIR"
        git pull origin main || true
        echo "✓ Repository updated"
    fi
    
    # Change to installation directory
    cd "$INSTALL_DIR"
fi

# 6. Detect GPU and configure environment
echo ""
echo "=== Configuring Environment ==="

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)

echo "Detected $GPU_COUNT GPU(s) with ${GPU_MEMORY}MB memory each"

# Auto-select profile based on GPU
if [ "$GPU_MEMORY" -ge 40000 ] && [ "$GPU_COUNT" -eq 1 ]; then
    echo "→ Using dev32 profile (single high-memory GPU)"
    cp .env.dev32 .env
elif [ "$GPU_COUNT" -ge 4 ]; then
    echo "→ Using prod480 profile (multi-GPU setup)"
    cp .env.prod480 .env
else
    echo "→ Using dev32 profile (default)"
    cp .env.dev32 .env
fi

# Auto-generate CUDA_DEVICES based on GPU count
if [ "$GPU_COUNT" -gt 1 ]; then
    # Generate comma-separated list: 0,1,2,3...
    CUDA_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
    echo "✓ Setting CUDA_DEVICES=$CUDA_DEVICES"
    sed -i "s/CUDA_DEVICES=.*/CUDA_DEVICES=$CUDA_DEVICES/" .env
fi

# Set workspace to home directory
if [ -n "${WORKSPACE_PATH:-}" ]; then
    sed -i "s|WORKSPACE_HOST=.*|WORKSPACE_HOST=${WORKSPACE_PATH}|" .env
    echo "✓ Workspace set to: ${WORKSPACE_PATH}"
else
    sed -i "s|WORKSPACE_HOST=.*|WORKSPACE_HOST=$HOME|" .env
    echo "✓ Workspace set to: $HOME"
fi

# 7. Create necessary directories
echo ""
echo "=== Creating Directories ==="
mkdir -p models cache workspace logs
chmod 755 models cache workspace logs
echo "✓ Directories created"

# 8. Pull Docker images
echo ""
echo "=== Pulling Docker Images ==="
docker-compose pull
echo "✓ Images pulled"

# 9. Build custom images
echo ""
echo "=== Building Custom Images ==="
docker-compose build
echo "✓ Custom images built"

# 10. Start services
echo ""
echo "=== Starting Services ==="
docker-compose up -d

# 11. Wait for services to be ready
echo ""
echo "=== Waiting for Services ==="
sleep 10

# Check vLLM
wait_for_service "vLLM" "http://localhost:8000/health"

# 12. Download model if needed
echo ""
echo "=== Model Management ==="
MODEL_NAME=$(grep "MODEL_NAME=" .env | cut -d'=' -f2)
MODEL_DIR="models/$(echo $MODEL_NAME | tr '/' '_')"
echo "Model: $MODEL_NAME"

# Check if model needs to be downloaded
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "Model not found locally."
    echo ""
    
    # Prompt for download
    if [ "${AUTO_DOWNLOAD_MODEL:-true}" = "true" ]; then
        echo "Starting automatic model download (approximately 65GB)..."
        echo "This may take 30-60 minutes depending on your connection speed."
        echo ""
        
        # Create models directory
        mkdir -p models
        
        # Download using docker
        docker run --rm \
            -v $(pwd)/models:/models \
            -e HF_HOME=/models \
            -e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-} \
            vllm/vllm-openai:latest \
            python -c "
from huggingface_hub import snapshot_download
import os
model_name = '$MODEL_NAME'
local_dir = '/models/$(echo $MODEL_NAME | tr '/' '_')'
print(f'Downloading {model_name} to {local_dir}...')
try:
    snapshot_download(model_name, local_dir=local_dir, local_dir_use_symlinks=False)
    print('✓ Model downloaded successfully!')
except Exception as e:
    print(f'✗ Download failed: {e}')
    exit(1)
"
        
        if [ $? -eq 0 ]; then
            echo "✓ Model download complete"
        else
            echo "✗ Model download failed"
            echo "You can retry with: make fetch-models"
            echo "Continuing without model..."
        fi
    else
        echo "Automatic download disabled. To download manually:"
        echo "  make fetch-models"
        echo ""
        echo "Or set AUTO_DOWNLOAD_MODEL=true and run this script again"
    fi
else
    echo "✓ Model already present"
    # Check size to ensure it's complete
    size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
    echo "  Size: $size"
fi

# 13. Show service status
echo ""
echo "=== Service Status ==="
docker-compose ps

# 14. Test API
echo ""
echo "=== Testing API ==="
if command_exists python3; then
    python3 test_api.py || echo "API test requires model to be downloaded first"
else
    echo "Python not found, skipping API test"
fi

# 15. Show access information
echo ""
echo "==========================================="
echo "✓ Stack Setup Complete!"
echo "==========================================="
echo ""
echo "Access Points:"
echo "  - vLLM API: http://localhost:8000"
echo "  - OpenAI API: http://localhost:8000/v1"
echo "  - Caddy (HTTP): http://localhost:80"
echo ""
echo "Workspace Monitoring:"
echo "  - Path: ${WORKSPACE_PATH:-$HOME}"
echo "  - Files are monitored for code-aware responses"
echo ""
echo "Useful Commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Check health: make health"
echo "  - Test API: make test"
echo "  - Stop services: make stop"
echo "  - Download model: make fetch-models"
echo ""
echo "Configuration:"
echo "  - GPU: $GPU_COUNT x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "  - Model: $MODEL_NAME"
echo "  - Profile: $(basename .env)"
echo ""

# 16. Keep container running (if needed)
if [ "${KEEP_RUNNING:-false}" = "true" ]; then
    echo "Container will keep running. Press Ctrl+C to stop."
    # Monitor services and restart if needed
    while true; do
        sleep 60
        # Check if vLLM is still healthy
        if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM appears down, restarting services..."
            docker-compose restart vllm
        fi
    done
else
    echo "Setup complete. Services are running in background."
fi