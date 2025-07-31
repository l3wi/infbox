# Multi-GPU Code-Aware LLM Inference Stack Implementation Plan

## Overview
This project implements a production-ready inference stack for large language models with code-aware capabilities, featuring:
- Multi-GPU support with configurable presets (dev32 for single GPU, prod480 for multi-GPU)
- Real-time code indexing with KV cache prefetching
- OpenAI-compatible API endpoint via vLLM
- Automatic file watching with .gitignore support
- Easy deployment with single bootstrap script

## Architecture Summary
- **vLLM**: Serves the LLM with OpenAI-compatible API
- **LMCache**: Manages KV cache with FP16 precision
- **Cake**: Adaptive KV chunk streaming
- **Watcher**: inotify-based file monitoring
- **Caddy**: Reverse proxy with automatic TLS

## Implementation Status

### Completed Tasks

1. ✅ **Project Structure Setup**
   - Created all directories and base files
   - Set up proper file permissions
   - Added .gitignore for clean repository

2. ✅ **Bootstrap Script** (bootstrap.sh)
   - GPU detection and profile selection
   - Docker and NVIDIA toolkit checks
   - Automatic environment setup based on GPU

3. ✅ **Docker Compose Configuration**
   - Base compose file with all 5 services
   - Environment variable substitution
   - Health checks and restart policies
   - Proper networking and volumes

4. ✅ **File Watcher Service**
   - Python-based implementation with watchdog
   - .gitignore parsing and respect
   - xxhash-based deduplication
   - Real-time updates to LMCache

5. ✅ **Environment Configurations**
   - .env.dev32 for A6000/single GPU
   - .env.prod480 for multi-GPU setup
   - Proper quantization settings

6. ✅ **Makefile**
   - Easy commands for dev/prod deployment
   - Health checking and log viewing
   - Service management utilities

7. ✅ **Model Fetching Utility**
   - Automated HuggingFace model downloads
   - Progress tracking and verification
   - Resume support for large models

8. ✅ **TLS Configuration**
   - Caddyfile with automatic HTTPS
   - Reverse proxy to vLLM
   - Security headers

9. ✅ **Documentation**
   - Comprehensive README
   - Quick start guide
   - API usage examples

### Deployment Ready

The system is now ready for deployment on your A6000 server. Simply:

1. SSH to your server
2. Clone this repository
3. Run `./bootstrap.sh`
4. Run `make start-dev`

The A6000 with 48GB VRAM is perfect for the dev32 profile with Qwen2.5-Coder-32B.

## Implementation Tasks

### Phase 1: Core Infrastructure (Days 1-3)

#### 1. Project Structure Setup
```
infbox/
├── bootstrap.sh          # One-click setup script
├── Makefile             # Build/deploy commands
├── docker-compose.yml   # Base compose file
├── docker-compose.prod.yml # Production overrides
├── .env.dev32          # Dev environment (32B model)
├── .env.prod480        # Prod environment (480B model)
├── Caddyfile           # TLS configuration
├── scripts/
│   ├── watch_codebase.py   # File watcher
│   ├── fetch_models.sh     # Model download utility
│   └── health_check.py     # Service health monitoring
├── monitoring/         # Grafana dashboards
│   └── dashboard.json
└── docs/
    └── README.md
```

#### 2. Bootstrap Script
- Install NVIDIA drivers, Docker, Docker Compose
- Clone repository and setup directories
- Configure environment based on GPU detection
- Initialize model download
- Start services

#### 3. Docker Compose Configuration
- Base compose file with all services
- Environment variable substitution for MODEL_NAME, GPU_COUNT, PREC
- Volume mounts for models, cache, and workspace
- Network configuration for inter-service communication

### Phase 2: Core Services (Days 4-6)

#### 4. File Watcher Service
- Python-based inotify watcher
- .gitignore parsing and filtering
- Hash-based deduplication
- Real-time updates to LMCache
- Connection to vLLM endpoint

#### 5. Model Management
- Automated model fetching based on preset
- Support for GPTQ-Int4 and FP8 variants
- Progress tracking and resumable downloads
- Model verification and loading

#### 6. Service Integration
- LMCache configuration with local CPU/disk caching
- Cake loader with adaptive streaming
- vLLM with KV transfer configuration
- Health checks and restart policies

### Phase 3: Deployment & Operations (Days 7-9)

#### 7. Monitoring Setup
- Grafana dashboard with:
  - GPU utilization metrics
  - KV cache hit rates
  - Request latency
  - Model throughput
- Prometheus metrics exporter

#### 9. Documentation
- Comprehensive README
- Upgrade guide between presets
- Troubleshooting guide
- API usage examples

## Technical Decisions & Rationale

### 1. KV Cache Always FP16
- Maintains quality while allowing weight quantization
- Critical for code-aware tasks requiring precision
- Balanced memory usage with performance

### 2. Configurable Presets
- dev32: Single GPU development (24-48GB VRAM)
  - Qwen2.5-Coder-32B-Instruct-GPTQ-Int4
  - Suitable for RTX 4090 or single A100
- prod480: Multi-GPU production (4+ A100 80GB)
  - Qwen3-Coder-480B-A35B-Instruct-FP8
  - Full capability deployment

### 3. Container Architecture
- Each service in isolated container
- Clear separation of concerns
- Easy scaling and updates
- Standardized logging and monitoring

### 4. File Watching Strategy
- inotify for efficient file system monitoring
- Respect .gitignore patterns
- Incremental updates only
- Hash-based deduplication

## MVP Scope
- Focus on single-node deployment first
- Basic authentication (API key)
- Essential monitoring only
- Manual backup/restore
- Single workspace support

## Future Enhancements (Post-MVP)
- Multi-node GPU clustering with P2P
- Redis backend for distributed caching
- Advanced authentication (OAuth/SAML)
- Multi-workspace support
- Automated backups
- AWQ quantization support

## Success Criteria
1. One-command deployment from fresh Ubuntu 22.04
2. < 5 minute setup time (excluding model download)
3. Seamless dev to prod transition via env file
4. Real-time code indexing with < 1s latency
5. OpenAI-compatible API with full feature parity

## Risk Mitigation
- Fallback to CPU inference for testing
- Graceful degradation without GPU
- Model download retry logic
- Service health monitoring
- Automatic restart on failure

This plan provides a clear path from development to production deployment while maintaining simplicity and reliability.