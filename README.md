# Multi-GPU Code-Aware LLM Inference Stack

A production-ready inference stack for large language models with real-time code indexing and KV cache optimization.

## Features

- 🚀 **One-command deployment** on Ubuntu 22.04 with NVIDIA GPUs
- 🔄 **Dual configuration profiles**: dev32 (single GPU) and prod480 (multi-GPU)
- 📁 **Real-time code indexing** with .gitignore support
- 🔌 **OpenAI-compatible API** via vLLM
- 💾 **Optimized KV caching** with LMCache (always FP16)
- 🔒 **Automatic TLS** with Caddy
- 📊 **Built-in monitoring** capabilities

## Quick Start

### Option 1: One-Line Installation
```bash
# Run this on your GPU server:
curl -sSL https://raw.githubusercontent.com/l3wi/infbox/main/quick_start.sh | sudo bash
```

### Option 2: Standard Deployment
```bash
git clone https://github.com/l3wi/infbox
cd infbox
./bootstrap.sh
make start-dev  # For single GPU
# or
make start-prod # For multi-GPU
```

### Option 3: Automated Container Deployment
```bash
# Deploy everything in one command
./deploy.sh native

# Or run in Docker-in-Docker mode
./deploy.sh docker --workspace /path/to/code

# Or deploy to remote server
./deploy.sh ssh --host user@server.com
```

### Option 4: Cloud/Server Provisioning
When provisioning a new server, use the container start script:
```bash
# This script handles everything automatically
./container_start.sh
```

The container start script will:
- Install Docker and NVIDIA drivers
- Detect GPU and select appropriate profile
- Pull and build all images
- Start all services
- Test the API
- Show access information

## Requirements

- Ubuntu 22.04
- NVIDIA GPU(s):
  - Dev: 1x A6000 (48GB), A100 (40GB+), or RTX 4090 (24GB)
  - Prod: 4x A100 (80GB) or equivalent
- CUDA 12.0+
- Docker & Docker Compose

## Configuration

The stack supports two presets via environment files:

### Development (.env.dev32)
- Model: Qwen2.5-Coder-32B-Instruct-GPTQ-Int4
- Single GPU with INT4 quantization
- Suitable for A6000 or RTX 4090

### Production (.env.prod480)
- Model: Qwen3-Coder-480B-A35B-Instruct-FP8
- Multi-GPU with FP8 quantization
- Requires 4+ A100 80GB GPUs

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────┐
│   Caddy     │────▶│  vLLM + LMCache  │◀────│  Watcher │
│  (TLS/Proxy)│     │   (Integrated)   │     │(File Mon)│
└─────────────┘     └──────────────────┘     └──────────┘
```

## Services

- **vLLM**: Main LLM inference server with integrated LMCache for KV caching
- **Watcher**: File system monitoring for code awareness
- **Caddy**: Reverse proxy with auto-TLS

## Commands

```bash
make start       # Start with current config
make start-dev   # Start development profile
make start-prod  # Start production profile
make stop        # Stop all services
make logs        # View all logs
make logs-vllm   # View vLLM logs only
make health      # Check service health
make clean       # Remove all containers/volumes
```

## API Usage

The stack provides an OpenAI-compatible API:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not required for local deployment
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    messages=[
        {"role": "user", "content": "Write a Python function to sort a list"}
    ]
)
```

## File Watching

The watcher service automatically:
- Monitors your home directory by default (`~/`)
- Respects .gitignore patterns
- Updates KV cache on file changes
- Deduplicates using xxhash

To change the monitored directory, edit `WORKSPACE_HOST` in `.env`:
```bash
# Monitor home directory (default)
WORKSPACE_HOST=~/

# Monitor specific project
WORKSPACE_HOST=~/projects/myapp

# Monitor absolute path
WORKSPACE_HOST=/opt/code
```

## Troubleshooting

### GPU not detected
```bash
nvidia-smi  # Check GPU availability
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Service health check
```bash
make health
curl http://localhost:8000/health
```

### View logs
```bash
make logs
docker-compose logs -f vllm
```

## License

[Your License Here]