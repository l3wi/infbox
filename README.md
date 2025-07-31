# InfBox - Multi-GPU Code-Aware LLM Inference Stack

One-command setup for high-performance LLM inference with automatic GPU detection and model management.

## Quick Start

```bash
git clone https://github.com/yourusername/infbox.git
cd infbox
./bootstrap.sh
```

That's it! The bootstrap script will:
1. Check system requirements (Ubuntu 22.04, NVIDIA drivers)
2. Install Docker and NVIDIA Container Toolkit if needed
3. Detect your GPU and select the appropriate profile
4. Download the model (Qwen2.5-Coder-32B-AWQ for single GPU)
5. Start the inference stack

## Features

- **Auto GPU Detection**: Automatically selects between dev (single GPU) and production (multi-GPU) profiles
- **One-Command Setup**: Complete environment setup, model download, and service launch
- **OpenAI Compatible**: Drop-in replacement for OpenAI API
- **Code-Aware**: Watches your workspace and maintains context
- **Production Ready**: Includes health checks, auto-restart, and TLS support

## Supported Configurations

### Dev Profile (dev32)
- **GPU**: Single A6000 (48GB) or 4090 (24GB)
- **Model**: Qwen2.5-Coder-32B-Instruct-AWQ
- **Context**: 8K tokens
- **Quantization**: INT4 (AWQ)

### Production Profile (prod480)
- **GPU**: 4+ A100 80GB
- **Model**: Qwen3-Coder-480B-Instruct-FP8
- **Context**: 131K tokens
- **Quantization**: FP8

## API Endpoints

- **vLLM API**: http://localhost:8000
- **OpenAI Compatible**: http://localhost:8000/v1
- **Health Check**: http://localhost:8000/health

## Usage

### Chat Completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### View Logs
```bash
docker compose logs -f vllm
```

### Stop Services
```bash
docker compose down
```

### Restart Services
```bash
docker compose restart
```

## Environment Variables

Key variables in `.env`:
- `MODEL_NAME`: Model to use
- `GPU_COUNT`: Number of GPUs
- `MAX_MODEL_LEN`: Maximum context length
- `WORKSPACE_HOST`: Directory to watch (default: ~/)

## Requirements

- Ubuntu 22.04
- NVIDIA GPU with CUDA support
- 24GB+ VRAM (dev) or 320GB+ VRAM (production)
- 50GB+ disk space for models

## Architecture

The stack consists of:
- **vLLM**: High-performance inference engine
- **Watcher**: Monitors code changes in your workspace
- **Caddy**: Reverse proxy with automatic HTTPS

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Check NVIDIA drivers
```

### Docker Permission Issues
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Model Download Failed
Check disk space and network connection. Resume download by running:
```bash
./bootstrap.sh
```

## License

MIT