#!/bin/bash
set -euo pipefail

# Model fetching utility for Multi-GPU LLM Inference Stack
# Downloads models based on current environment configuration

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "ERROR: .env file not found. Run bootstrap.sh first."
    exit 1
fi

# Model directory
MODEL_DIR="${MODEL_DIR:-./models}"
mkdir -p "$MODEL_DIR"

# Hugging Face CLI check
check_hf_cli() {
    if ! command -v huggingface-cli &> /dev/null; then
        echo "Installing Hugging Face CLI..."
        pip install -U huggingface-hub
    fi
}

# Download model based on configuration
download_model() {
    local model_name="$1"
    local model_path="$MODEL_DIR/$(echo $model_name | tr '/' '_')"
    
    if [ -d "$model_path" ]; then
        echo "Model already exists at $model_path"
        read -p "Re-download? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    echo "Downloading model: $model_name"
    echo "Target directory: $model_path"
    
    # Use huggingface-cli for download
    huggingface-cli download "$model_name" \
        --local-dir "$model_path" \
        --local-dir-use-symlinks False \
        --resume-download
    
    echo "Model downloaded successfully"
}

# Verify model files
verify_model() {
    local model_name="$1"
    local model_path="$MODEL_DIR/$(echo $model_name | tr '/' '_')"
    
    echo "Verifying model at $model_path..."
    
    # Check for essential files
    local required_files=(
        "config.json"
        "tokenizer_config.json"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$model_path/$file" ]; then
            echo "ERROR: Missing required file: $file"
            return 1
        fi
    done
    
    # Check model size
    local size=$(du -sh "$model_path" | cut -f1)
    echo "Model size: $size"
    
    return 0
}

# Main execution
main() {
    echo "=== Model Fetching Utility ==="
    echo "Current model: $MODEL_NAME"
    echo "Precision: $PREC"
    echo ""
    
    # Check dependencies
    check_hf_cli
    
    # Download model
    download_model "$MODEL_NAME"
    
    # Verify download
    if verify_model "$MODEL_NAME"; then
        echo ""
        echo "✓ Model ready for use"
        echo ""
        echo "Model location: $MODEL_DIR/$(echo $MODEL_NAME | tr '/' '_')"
        echo "You can now start the stack with 'make start'"
    else
        echo ""
        echo "✗ Model verification failed"
        echo "Please check the download and try again"
        exit 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -m, --model    Override model from .env"
    echo "  -d, --dir      Override model directory"
    echo ""
    echo "Examples:"
    echo "  $0                    # Use model from .env"
    echo "  $0 -m Qwen/Qwen2.5-Coder-32B-Instruct"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -d|--dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main