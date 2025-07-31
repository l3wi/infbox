#!/bin/bash
set -euo pipefail

# Quick Start Script for LLM Inference Stack
# Can be run directly: curl -sSL https://raw.githubusercontent.com/l3wi/infbox/main/quick_start.sh | bash

echo "==========================================="
echo "LLM Inference Stack - Quick Start"
echo "==========================================="
echo ""

# Configuration
GIT_REPO="https://github.com/l3wi/infbox"
INSTALL_DIR="${INSTALL_DIR:-/opt/infbox}"

# Ensure running with appropriate permissions
if [ "$EUID" -ne 0 ] && [ ! -w "/opt" ]; then 
    echo "This script needs to install to /opt. Please run with sudo:"
    echo "  curl -sSL https://raw.githubusercontent.com/l3wi/infbox/main/quick_start.sh | sudo bash"
    exit 1
fi

# Install git if not present
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    apt-get update -qq
    apt-get install -y -qq git
fi

# Clone or update repository
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Cloning repository..."
    git clone "$GIT_REPO" "$INSTALL_DIR"
else
    echo "Updating repository..."
    cd "$INSTALL_DIR"
    git pull origin main
fi

# Change to installation directory
cd "$INSTALL_DIR"

# Make scripts executable
chmod +x container_start.sh bootstrap.sh deploy.sh scripts/*.sh

# Run the container start script
echo ""
echo "Starting automatic setup..."
./container_start.sh

echo ""
echo "==========================================="
echo "Quick start complete!"
echo "Installation directory: $INSTALL_DIR"
echo "==========================================="