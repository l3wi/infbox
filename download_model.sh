#!/bin/bash
set -euo pipefail

# Native model download script for LLM Inference Stack
# Downloads models directly to host filesystem without Docker

echo "=== Model Download Utility ==="
echo ""

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "No .env file found. Using dev32 profile."
    cp .env.dev32 .env
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-32B-Instruct}"
MODELS_PATH="${MODELS_PATH:-./models}"
MODEL_DIR="$MODELS_PATH/$(echo $MODEL_NAME | tr '/' '_')"

echo "Model: $MODEL_NAME"
echo "Download path: $MODEL_DIR"
echo ""

# Check if model already exists
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    size=$(du -sh "$MODEL_DIR" | cut -f1)
    echo "Model already exists (size: $size)"
    read -p "Re-download? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Download cancelled."
        exit 0
    fi
    echo "Removing existing model..."
    rm -rf "$MODEL_DIR"
fi

# Create models directory
mkdir -p "$MODELS_PATH"

# Check Python and pip
if ! command -v python3 &> /dev/null; then
    echo "Python3 is required. Installing..."
    apt-get update && apt-get install -y python3 python3-pip
fi

# Install huggingface-hub if needed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip3 install -U huggingface-hub
fi

# Download model
echo ""
echo "Starting download (this may take 30-60 minutes)..."
echo "Model size: approximately 65GB"
echo ""

# Create Python download script
cat > /tmp/download_model.py << 'EOF'
import os
import sys
from huggingface_hub import snapshot_download

model_name = sys.argv[1]
local_dir = sys.argv[2]
token = os.environ.get('HF_TOKEN', None)

print(f"Downloading {model_name}...")
print(f"Destination: {local_dir}")

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
    print("\n✗ Download interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    sys.exit(1)
EOF

# Run download
python3 /tmp/download_model.py "$MODEL_NAME" "$MODEL_DIR"

# Cleanup
rm -f /tmp/download_model.py

# Verify download
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A $MODEL_DIR)" ]; then
    echo ""
    echo "=== Download Complete ==="
    echo "Model location: $MODEL_DIR"
    echo "Size: $(du -sh "$MODEL_DIR" | cut -f1)"
    echo ""
    echo "You can now start the inference stack with:"
    echo "  make start"
else
    echo ""
    echo "✗ Download verification failed"
    exit 1
fi