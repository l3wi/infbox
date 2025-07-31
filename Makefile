# InfBox - Multi-GPU Code-Aware LLM Inference Stack
.PHONY: help start stop restart logs clean test setup

# Default target
help:
	@echo "InfBox Management Commands:"
	@echo "  make setup      - Run bootstrap script to set up everything"
	@echo "  make start      - Start standard inference stack"
	@echo "  make start-opt  - Start optimized stack (hierarchical caching)"
	@echo "  make start-lm   - Start LMCache stack (individual file caching)"
	@echo "  make stop       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - View logs (all services)"
	@echo "  make logs-vllm  - View vLLM logs"
	@echo "  make logs-watch - View watcher logs"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean up containers and volumes"
	@echo "  make status     - Show service status"

# Setup
setup:
	@./bootstrap.sh

# Start services
start:
	@docker compose up -d
	@echo "✅ Standard stack started"
	@echo "   API: http://localhost:8000"

start-opt:
	@docker compose -f config/docker-compose.optimized.yml up -d
	@echo "✅ Optimized stack started (hierarchical caching)"
	@echo "   API: http://localhost:8000"

start-lm:
	@docker compose -f config/docker-compose.lmcache.yml up -d
	@echo "✅ LMCache stack started (individual file caching)"
	@echo "   API: http://localhost:8000"

# Stop services
stop:
	@docker compose down
	@docker compose -f config/docker-compose.optimized.yml down 2>/dev/null || true
	@docker compose -f config/docker-compose.lmcache.yml down 2>/dev/null || true
	@echo "✅ All services stopped"

# Restart services
restart: stop start

# View logs
logs:
	@docker compose logs -f

logs-vllm:
	@docker compose logs -f vllm

logs-watch:
	@docker compose logs -f watcher

# Test
test:
	@echo "Running caching tests..."
	@python scripts/tests/test_prefix_caching.py
	@python scripts/tests/test_lmcache_access.py

test-api:
	@curl -s http://localhost:8000/health || echo "❌ API not responding"
	@curl -s http://localhost:8000/v1/models | jq . || echo "❌ Models endpoint failed"

# Clean up
clean:
	@docker compose down -v
	@docker compose -f config/docker-compose.optimized.yml down -v 2>/dev/null || true
	@docker compose -f config/docker-compose.lmcache.yml down -v 2>/dev/null || true
	@echo "✅ Cleaned up containers and volumes"

clean-models:
	@echo "⚠️  This will delete downloaded models!"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/*; \
		echo "✅ Models deleted"; \
	fi

# Status
status:
	@echo "Service Status:"
	@docker compose ps
	@echo ""
	@echo "Cache Statistics:"
	@docker compose logs watcher | grep "Cache stats" | tail -1 || echo "No cache stats available"

# Development
shell-vllm:
	@docker compose exec vllm bash

shell-watcher:
	@docker compose exec watcher bash

# Quick health check
health:
	@curl -s http://localhost:8000/health && echo " ✅ vLLM healthy" || echo " ❌ vLLM not responding"