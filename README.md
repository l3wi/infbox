# InfBox - Multi-GPU Code-Aware LLM Inference Stack

One-command setup for high-performance LLM inference with automatic GPU detection, model management, and integrated LMCache support.

## Quick Start

### One-line install (Qwen3-Coder-480B):
```bash
curl -L https://raw.githubusercontent.com/l3wi/infbox/main/install.sh | bash
```

### Or clone and setup manually:
```bash
git clone https://github.com/l3wi/infbox.git
cd infbox
make setup
```

That's it! The bootstrap script will:
1. Check system requirements (Ubuntu 22.04, NVIDIA drivers)
2. Install Docker and NVIDIA Container Toolkit if needed
3. Detect your GPU and select the appropriate profile
4. Download the model (Qwen2.5-Coder-32B-AWQ for single GPU)
5. Start the inference stack with vLLM and LMCache integration

## Features

- **Auto GPU Detection**: Automatically selects between dev (single GPU) and production (multi-GPU) profiles
- **One-Command Setup**: Complete environment setup, model download, and service launch
- **OpenAI Compatible**: Drop-in replacement for OpenAI API
- **Code-Aware with Proactive Caching**: Watches your workspace and proactively caches files in vLLM for faster context switching
- **LMCache Integration**: Efficient KV cache management for improved performance
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

### Standard Setup (Recommended)
```bash
make start                # Basic caching
```

### Optimized Setup (Better Performance)
```bash
make start-opt           # Hierarchical prefix caching
```

### LMCache Setup (Best for Large Codebases)
```bash
make start-lm            # Individual file caching (handles thousands of files)
```

### Other Commands
```bash
make stop                # Stop all services
make restart             # Restart services
make logs                # View all logs
make logs-vllm          # View vLLM logs only
make logs-watch         # View watcher logs only
make test               # Run tests
make status             # Show service status
make clean              # Clean up containers
```

### Chat Completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
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

## License

MIT