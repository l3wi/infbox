# InfBox - Multi-GPU Code-Aware LLM Inference Stack
.PHONY: help setup start stop restart logs clean status health

# Default target
help:
	@echo "InfBox Commands:"
	@echo "  make setup    - Run bootstrap script (auto-setup everything)"
	@echo "  make start    - Start services"
	@echo "  make stop     - Stop services"
	@echo "  make restart  - Restart services"
	@echo "  make logs     - View logs (all services)"
	@echo "  make status   - Show service status"
	@echo "  make health   - Quick health check"
	@echo "  make clean    - Clean up containers"

# One-command setup
setup:
	@./bootstrap.sh

# Service management
start:
	@docker compose up -d
	@echo "✅ InfBox started"
	@echo "   API: http://localhost:8000"

stop:
	@docker compose down
	@echo "✅ Services stopped"

restart: stop start

# Logs
logs:
	@docker compose logs -f

logs-vllm:
	@docker compose logs -f vllm

logs-watcher:
	@docker compose logs -f watcher

# Status and health
status:
	@docker compose ps

health:
	@curl -s http://localhost:8000/health && echo " ✅ vLLM healthy" || echo " ❌ vLLM not responding"

# Cleanup
clean:
	@docker compose down -v
	@echo "✅ Cleaned up containers and volumes"

clean-models:
	@echo "⚠️  This will delete downloaded models!"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/*; \
		echo "✅ Models deleted"; \
	fi