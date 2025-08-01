# InfBox - Multi-GPU Code-Aware LLM Inference Stack

One-command setup for Qwen3-Coder-480B with optimized vLLM configuration for maximum performance.

## Quick Start

### One-line install:
```bash
curl -L https://raw.githubusercontent.com/l3wi/infbox/main/install.sh | bash
```

This will:
- Check for 140GB+ VRAM and NVIDIA drivers
- Auto-install Docker and NVIDIA Container Toolkit
- Download Qwen3-Coder-480B-FP8 model (~200GB)
- Configure optimized vLLM settings
- Start inference server with 238K token context

## Requirements

- **GPU**: Minimum 140GB total VRAM (e.g., 2× H100 80GB)
- **OS**: Ubuntu 20.04+ or similar Linux
- **Disk**: 500GB+ free space
- **Network**: Fast connection for model download

## Model Configuration

**Model**: Qwen3-Coder-480B-A35B-Instruct-FP8
- **Quantization**: FP8 with bfloat16 compute
- **Context Length**: 238,000 tokens
- **GPU Utilization**: 97%
- **KV Cache**: FP8 optimized
- **Special Features**: 
  - VLLM v1 architecture
  - Tool calling support
  - Optimized for H100/H200 GPUs

## Features

- **Automatic Installation**: Installs all dependencies including Docker and NVIDIA toolkit
- **Optimized Performance**: Pre-configured for maximum throughput
- **OpenAI Compatible**: Drop-in replacement for OpenAI API
- **Code-Aware**: Workspace monitoring for context-aware responses
- **Production Ready**: Health checks, auto-restart, and monitoring

## API Endpoints

- **vLLM API**: http://localhost:8000
- **OpenAI Compatible**: http://localhost:8000/v1
- **Health Check**: http://localhost:8000/health

## Usage

### Test the API
```bash
# Check if model is loaded
curl http://localhost:8000/v1/models

# Simple chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "messages": [{"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}],
    "max_tokens": 500
  }'
```

### Long Context Example (up to 238K tokens)
```bash
# Review large codebase
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "messages": [
      {"role": "system", "content": "You are an expert code reviewer."},
      {"role": "user", "content": "Review this codebase and suggest improvements: [paste your code here]"}
    ],
    "max_tokens": 4096,
    "temperature": 0.7
  }'
```

### Service Management
```bash
cd ~/infbox
docker compose logs -f          # View logs
docker compose restart          # Restart services
docker compose down             # Stop services
docker compose up -d            # Start services
```

## Environment Variables

Key variables in `.env`:
- `MODEL_NAME`: Qwen3-Coder-480B-A35B-Instruct-FP8
- `GPU_COUNT`: Number of GPUs for tensor parallelism
- `GPU_UTIL`: GPU memory utilization (0.97 = 97%)
- `MAX_MODEL_LEN`: Maximum context length (238000)
- `DTYPE`: Compute dtype (bfloat16)
- `QUANTIZATION`: Model quantization (fp8)
- `KV_CACHE_DTYPE`: KV cache precision (fp8)
- `VLLM_USE_V1`: Use vLLM v1 architecture
- `TORCH_CUDA_ARCH_LIST`: CUDA architecture (9.0 for H100/H200)

## Project Structure

```
infbox/
├── bootstrap.sh           # One-command setup script
├── docker-compose.yml     # Main service configuration
├── Makefile              # Easy command shortcuts
├── config/               # Configuration files
│   ├── Caddyfile         # Reverse proxy config
│   ├── docker-compose.lmcache.yml     # LMCache variant
│   └── docker-compose.optimized.yml   # Optimized variant
├── docker/               # Docker build files
│   └── Dockerfile.*      # Various container definitions
├── scripts/              # Utility scripts
│   ├── tests/           # Test scripts
│   └── watchers/        # File watching implementations
├── docs/                # Documentation
├── models/              # Downloaded models (gitignored)
├── cache/               # Runtime cache (gitignored)
└── logs/                # Service logs (gitignored)
```

## Architecture

The stack consists of:
- **vLLM**: High-performance inference engine with automatic prefix caching for efficient KV cache reuse
- **Optimized Watcher**: Intelligently organizes code context for maximum cache efficiency
- **Caddy**: Reverse proxy with automatic HTTPS

### Caching Strategy

We use an optimized single-instance approach that maximizes vLLM's automatic prefix caching:

- **Hierarchical Context**: Files organized by importance (Core → Frequent → Recent)
- **Stable Prefixes**: Most important files always appear first for consistent caching
- **Smart Scoring**: Entry points and frequently accessed files get priority
- **Performance**: 2-5x faster responses for queries with shared context

For details, see [CACHING_STRATEGY.md](CACHING_STRATEGY.md)

### Watcher Features
- **Intelligent File Scoring**: Prioritizes main files, configs, and frequently used code
- **Hierarchical Organization**: Three-layer context structure for optimal cache reuse
- **Smart Filtering**: Only processes code files (.py, .js, .ts, .go, etc.) and respects .gitignore
- **Batch Updates**: Minimizes cache invalidation with cooldown period
- **Change Detection**: Uses xxhash for fast content change detection

## Current Status

✅ **Working Components:**
- Bootstrap script successfully sets up the environment
- vLLM running with Qwen2.5-Coder-32B-Instruct-AWQ model
- LMCache integrated with vLLM for KV cache management
- Enhanced watcher with proactive file caching
- Streamlined repository with single bootstrap entry point
- Clean docker-compose setup with all services

🔧 **Configuration:**
- Model path: `/models/Qwen_Qwen2.5-Coder-32B-Instruct-AWQ`
- vLLM endpoint: `http://localhost:8000`
- Workspace monitoring: `~/` (configurable via WORKSPACE_HOST)
- Cache batch size: 5 files
- Max file size for caching: 100KB

📝 **Recent Changes:**
- Consolidated multiple docker-compose files into single file
- Removed redundant watcher scripts
- Fixed model path issues in vLLM configuration
- Implemented proactive file caching in watcher
- Integrated LMCache with vLLM using official image

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

### Watcher Not Caching Files
Check watcher logs:
```bash
docker compose logs -f watcher
```

Ensure your workspace contains code files with supported extensions (.py, .js, .ts, etc.)

## Troubleshooting

### NVIDIA Container Toolkit Issues
If you get GPU access errors after installation:
```bash
sudo systemctl restart docker
# If that doesn't work, reboot the system
sudo reboot
```

### Model Download Issues
If the model download is interrupted:
```bash
# The script will resume from where it left off
curl -L https://raw.githubusercontent.com/l3wi/infbox/main/install.sh | bash
```

### Out of Memory Errors
If vLLM fails to start due to OOM:
1. Reduce `MAX_MODEL_LEN` in `.env`
2. Lower `GPU_UTIL` from 0.97 to 0.90
3. Ensure no other processes are using GPU memory

### Check GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Monitor continuously
```

## License

MIT