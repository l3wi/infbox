#!/bin/bash
set -euo pipefail

# Deployment script for LLM Inference Stack
# Supports various cloud providers and deployment methods

echo "=== LLM Inference Stack Deployment ==="
echo ""

# Function to show usage
usage() {
    echo "Usage: $0 [provider] [options]"
    echo ""
    echo "Providers:"
    echo "  docker    - Run everything in a single container (Docker-in-Docker)"
    echo "  native    - Run directly on the host (default)"
    echo "  ssh       - Deploy to remote server via SSH"
    echo ""
    echo "Options:"
    echo "  --workspace PATH    - Set workspace path (default: ~)"
    echo "  --repo URL         - Git repository URL"
    echo "  --host HOST        - SSH host for remote deployment"
    echo "  --key PATH         - SSH key path"
    echo ""
    echo "Examples:"
    echo "  $0 docker --workspace /home/user/code"
    echo "  $0 ssh --host user@server.com --key ~/.ssh/id_rsa"
    echo "  $0 native"
}

# Parse arguments
PROVIDER="${1:-native}"
shift || true

WORKSPACE_PATH="$HOME"
GIT_REPO_URL=""
SSH_HOST=""
SSH_KEY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --workspace)
            WORKSPACE_PATH="$2"
            shift 2
            ;;
        --repo)
            GIT_REPO_URL="$2"
            shift 2
            ;;
        --host)
            SSH_HOST="$2"
            shift 2
            ;;
        --key)
            SSH_KEY="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Deploy based on provider
case $PROVIDER in
    docker)
        echo "=== Docker-in-Docker Deployment ==="
        echo "This will run the entire stack inside a single container"
        echo ""
        
        # Build the server image
        echo "Building server image..."
        docker build -f Dockerfile.server -t llm-inference-stack:latest .
        
        # Run with docker-compose
        echo "Starting server container..."
        HOST_WORKSPACE="$WORKSPACE_PATH" \
        GIT_REPO_URL="$GIT_REPO_URL" \
        docker-compose -f docker-compose.server.yml up -d
        
        echo ""
        echo "✓ Deployment complete!"
        echo "The entire stack is running inside the container."
        echo "Access the API at: http://localhost:8000"
        echo ""
        echo "To view container logs:"
        echo "  docker logs -f llm-inference-server"
        ;;
        
    native)
        echo "=== Native Deployment ==="
        echo "Running directly on this host"
        echo ""
        
        # Run the container start script directly
        WORKSPACE_PATH="$WORKSPACE_PATH" \
        GIT_REPO_URL="$GIT_REPO_URL" \
        ./container_start.sh
        ;;
        
    ssh)
        echo "=== SSH Remote Deployment ==="
        
        if [ -z "$SSH_HOST" ]; then
            echo "ERROR: --host is required for SSH deployment"
            usage
            exit 1
        fi
        
        SSH_OPTS=""
        if [ -n "$SSH_KEY" ]; then
            SSH_OPTS="-i $SSH_KEY"
        fi
        
        echo "Deploying to: $SSH_HOST"
        echo ""
        
        # Create temporary deployment package
        DEPLOY_PACKAGE="/tmp/llm-deploy-$(date +%s).tar.gz"
        echo "Creating deployment package..."
        tar -czf "$DEPLOY_PACKAGE" \
            --exclude='.git' \
            --exclude='models' \
            --exclude='cache' \
            --exclude='logs' \
            --exclude='*.tar.gz' \
            .
        
        # Copy to remote
        echo "Copying to remote server..."
        scp $SSH_OPTS "$DEPLOY_PACKAGE" "$SSH_HOST:/tmp/"
        
        # Execute deployment on remote
        echo "Executing remote deployment..."
        ssh $SSH_OPTS "$SSH_HOST" << 'ENDSSH'
            set -e
            
            # Create deployment directory
            mkdir -p ~/llm-inference-stack
            cd ~/llm-inference-stack
            
            # Extract package
            tar -xzf /tmp/llm-deploy-*.tar.gz
            rm /tmp/llm-deploy-*.tar.gz
            
            # Run deployment
            ./container_start.sh
ENDSSH
        
        # Cleanup
        rm "$DEPLOY_PACKAGE"
        
        echo ""
        echo "✓ Remote deployment complete!"
        echo "SSH to $SSH_HOST to manage the deployment"
        ;;
        
    *)
        echo "ERROR: Unknown provider: $PROVIDER"
        usage
        exit 1
        ;;
esac

echo ""
echo "=== Deployment Summary ==="
echo "Provider: $PROVIDER"
echo "Workspace: $WORKSPACE_PATH"
if [ -n "$GIT_REPO_URL" ]; then
    echo "Repository: $GIT_REPO_URL"
fi
echo ""
echo "Next steps:"
echo "1. Wait for model download (if needed): make fetch-models"
echo "2. Check service health: make health"
echo "3. Test the API: make test"