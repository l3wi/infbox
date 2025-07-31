.PHONY: help start start-dev start-prod stop logs clean fetch-models health

# Default target
help:
	@echo "Multi-GPU LLM Inference Stack"
	@echo ""
	@echo "Available commands:"
	@echo "  make start       - Start with current .env configuration"
	@echo "  make start-dev   - Start with dev32 profile (single GPU)"
	@echo "  make start-prod  - Start with prod480 profile (multi-GPU)"
	@echo "  make stop        - Stop all services"
	@echo "  make logs        - View logs (all services)"
	@echo "  make health      - Check service health"
	@echo "  make fetch-models - Download models"
	@echo "  make clean       - Clean up containers and volumes"

# Start with current configuration
start:
	@echo "Starting services with current configuration..."
	docker-compose up -d
	@echo "Services started. Check logs with 'make logs'"
	@echo "API available at http://localhost:8000"

# Start development profile
start-dev:
	@echo "Starting with dev32 profile..."
	cp .env.dev32 .env
	docker-compose up -d
	@echo "Dev stack started with Qwen2.5-Coder-32B"

# Start production profile  
start-prod:
	@echo "Starting with prod480 profile..."
	cp .env.prod480 .env
	docker-compose up -d
	@echo "Production stack started with Qwen3-Coder-480B"

# Stop all services
stop:
	@echo "Stopping all services..."
	docker-compose down
	@echo "Services stopped"

# View logs
logs:
	docker-compose logs -f

# View specific service logs
logs-%:
	docker-compose logs -f $*

# Health check
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health > /dev/null && echo "✓ vLLM is healthy" || echo "✗ vLLM is not responding"
	@docker-compose ps

# Fetch models
fetch-models:
	@echo "Fetching models based on current profile..."
	./scripts/fetch_models.sh

# Clean up
clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v
	@echo "Clean complete"

# Restart a specific service
restart-%:
	docker-compose restart $*

# Shell into a service
shell-%:
	docker-compose exec $* /bin/bash

# Show current configuration
config:
	@echo "Current configuration:"
	@cat .env

# Update services
update:
	docker-compose pull
	docker-compose up -d