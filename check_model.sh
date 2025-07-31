#!/bin/bash
# Check if Qwen model is downloaded

echo "=== Checking Model Status ==="
echo ""

# Check if models directory exists
if [ -d "models" ]; then
    echo "Models directory found"
    echo "Contents:"
    ls -la models/
    echo ""
    
    # Check for Qwen model
    if ls models/ | grep -q "Qwen"; then
        echo "✓ Qwen model directory found"
        # Check size
        if [ -d "models/Qwen_Qwen2.5-Coder-32B-Instruct" ]; then
            size=$(du -sh models/Qwen_Qwen2.5-Coder-32B-Instruct | cut -f1)
            echo "  Size: $size"
        fi
    else
        echo "✗ Qwen model not found"
        echo ""
        echo "To download the model, run:"
        echo "  make fetch-models"
        echo ""
        echo "Or manually download with:"
        echo "  docker run --rm -v ./models:/models vllm/vllm-openai:latest python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-32B-Instruct', local_dir='/models/Qwen_Qwen2.5-Coder-32B-Instruct')\""
    fi
else
    echo "✗ Models directory not found"
    echo "Creating models directory..."
    mkdir -p models
fi

echo ""
echo "Note: The 32B model is approximately 65GB in size"