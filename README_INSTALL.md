# InfBox Quick Install

One-line installation for Qwen3-Coder-480B inference stack:

```bash
curl -L https://raw.githubusercontent.com/l3wi/infbox/main/install.sh | bash
```

## What it does

1. **Checks system requirements**
   - NVIDIA drivers and GPUs (minimum 140GB VRAM)
   - Docker and NVIDIA Container Toolkit
   - Python3 and git

2. **Clones the repository** to `~/infbox`

3. **Downloads the model** to `~/models`
   - Qwen3-Coder-480B-A35B-Instruct-FP8 (~200GB)
   - Supports resume on interruption

4. **Configures the environment**
   - Sets up docker-compose for your GPU configuration
   - Optimizes settings for FP8 inference

5. **Starts the services**
   - vLLM inference server on port 8000
   - Workspace watcher for code context
   - Automatic health checks

6. **Creates instructions** at `~/INFBOX_README.txt`

## Requirements

- Ubuntu 20.04+ or similar Linux distribution
- NVIDIA GPUs with 140GB+ total VRAM
- Docker with NVIDIA Container Toolkit
- 500GB+ free disk space
- Python 3.8+

## Manual Installation

If you prefer to install step by step:

```bash
# Clone the repository
git clone https://github.com/l3wi/infbox.git
cd infbox

# Run the install script locally
./install.sh
```

## Configuration

The install script will:
- Auto-detect your GPU configuration
- Set up tensor parallelism for multi-GPU systems
- Configure optimal memory settings
- Create `.env` file with your specific setup

## After Installation

Check the created instructions file:
```bash
cat ~/INFBOX_README.txt
```

Test the API:
```bash
curl http://localhost:8000/v1/models
```

## Troubleshooting

If the installation fails:

1. Check GPU availability:
   ```bash
   nvidia-smi
   ```

2. Verify Docker GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

3. Check service logs:
   ```bash
   cd ~/infbox
   docker compose logs vllm
   ```

## Uninstall

To remove InfBox:

```bash
cd ~/infbox
docker compose down
cd ~
rm -rf infbox
rm -f INFBOX_README.txt
# Optionally remove model:
# rm -rf ~/models/Qwen_Qwen3-Coder-480B-A35B-Instruct-FP8
```