#!/bin/bash

# Quick status check script for LLM Inference Stack
# Can be run with: curl -sSL https://raw.githubusercontent.com/l3wi/infbox/main/check_status.sh | bash

echo "=== LLM Inference Stack Status Check ==="
echo "Time: $(date)"
echo ""

# Function to check endpoint
check_endpoint() {
    local name=$1
    local url=$2
    local expected=$3
    
    if curl -s -f -m 5 "$url" > /dev/null 2>&1; then
        echo "✓ $name is responding at $url"
        return 0
    else
        echo "✗ $name is NOT responding at $url"
        return 1
    fi
}

# Check Docker
echo "=== System Checks ==="
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed"
    docker --version
else
    echo "✗ Docker is NOT installed"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu,utilization.memory --format=csv,noheader | sed 's/^/  /'
else
    echo "✗ NVIDIA GPU not detected"
fi

echo ""

# Check services
echo "=== Service Status ==="

# Check if docker-compose is available and services are running
if command -v docker-compose &> /dev/null; then
    # Try to find the installation directory
    INSTALL_DIRS=("/opt/infbox" "$HOME/infbox" "." "/app")
    COMPOSE_DIR=""
    
    for dir in "${INSTALL_DIRS[@]}"; do
        if [ -f "$dir/docker-compose.yml" ]; then
            COMPOSE_DIR="$dir"
            break
        fi
    done
    
    if [ -n "$COMPOSE_DIR" ]; then
        echo "Found installation at: $COMPOSE_DIR"
        cd "$COMPOSE_DIR"
        echo ""
        echo "Running containers:"
        docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
    else
        echo "Docker Compose file not found in standard locations"
    fi
else
    echo "Docker Compose not available, checking containers directly:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(vllm|watcher|caddy|infbox)"
fi

echo ""

# Check endpoints
echo "=== API Endpoints ==="
check_endpoint "vLLM Health" "http://localhost:8000/health"
check_endpoint "vLLM Models" "http://localhost:8000/v1/models"
check_endpoint "Caddy HTTP" "http://localhost:80"

echo ""

# Quick API test
echo "=== API Test ==="
if curl -s -f "http://localhost:8000/health" > /dev/null 2>&1; then
    echo "Testing model list endpoint..."
    response=$(curl -s "http://localhost:8000/v1/models" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        echo "✓ API is responding"
        echo "Available models:"
        echo "$response" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | sed 's/^/  - /'
    else
        echo "✗ API returned empty response"
    fi
else
    echo "✗ API is not accessible"
fi

echo ""

# Memory usage
echo "=== Resource Usage ==="
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
fi

echo ""
echo "System Memory:"
free -h | grep -E "^Mem:" | awk '{print "  Used: " $3 " / Total: " $2}'

echo ""

# Logs preview
echo "=== Recent Logs (last 5 lines) ==="
if [ -n "$COMPOSE_DIR" ] && [ -f "$COMPOSE_DIR/docker-compose.yml" ]; then
    cd "$COMPOSE_DIR"
    echo "vLLM logs:"
    docker-compose logs --tail=5 vllm 2>/dev/null | sed 's/^/  /'
else
    echo "Unable to fetch logs - compose directory not found"
fi

echo ""
echo "=== Summary ==="
all_good=true

# Check critical services
if ! curl -s -f "http://localhost:8000/health" > /dev/null 2>&1; then
    all_good=false
    echo "⚠️  vLLM service is not healthy"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check if services are running: docker ps"
    echo "2. View logs: docker-compose logs -f vllm"
    echo "3. Restart services: cd /opt/infbox && docker-compose restart"
    echo "4. Check GPU: nvidia-smi"
else
    echo "✓ All services appear to be running correctly!"
    echo ""
    echo "Quick test command:"
    echo 'curl -X POST http://localhost:8000/v1/chat/completions \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -d '"'"'{"model": "Qwen/Qwen2.5-Coder-32B-Instruct",'
    echo '      "messages": [{"role": "user", "content": "Hello!"}],'
    echo '      "max_tokens": 50}'"'"
fi