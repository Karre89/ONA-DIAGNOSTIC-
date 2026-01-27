.PHONY: build up down logs seed ingest-sample test clean

# Docker commands
build:
	docker-compose -f infra/docker-compose.yml build

up:
	docker-compose -f infra/docker-compose.yml up --build

up-detached:
	docker-compose -f infra/docker-compose.yml up --build -d

down:
	docker-compose -f infra/docker-compose.yml down

logs:
	docker-compose -f infra/docker-compose.yml logs -f

logs-cloud:
	docker-compose -f infra/docker-compose.yml logs -f cloud

logs-edge:
	docker-compose -f infra/docker-compose.yml logs -f edge

# Database and seeding
seed:
	@echo "Seeding tenant, site, and device..."
	curl -X POST http://localhost:8000/api/v1/seed \
		-H "Content-Type: application/json" \
		-d '{"tenant_name": "Ona Kenya", "site_name": "Kenyatta Hospital", "site_code": "KNH001", "country": "KE", "device_name": "edge-box-001"}'

# Ingest sample study
ingest-sample:
	@echo "Triggering sample study ingest..."
	curl -X POST http://localhost:8080/api/ingest-sample \
		-H "Content-Type: application/json"

# Testing
test-cloud:
	cd cloud-platform && python -m pytest tests/ -v

test-edge:
	cd edge-agent && python -m pytest tests/ -v

test: test-cloud test-edge

# Acceptance tests
acceptance-test:
	@echo "Running acceptance tests..."
	@echo "1. Checking cloud health..."
	curl -f http://localhost:8000/health || exit 1
	@echo "\n2. Checking edge health..."
	curl -f http://localhost:8080/health || exit 1
	@echo "\n3. Checking results endpoint..."
	curl -f http://localhost:8000/api/v1/results/recent || exit 1
	@echo "\nAll acceptance tests passed!"

# Cleanup
clean:
	docker-compose -f infra/docker-compose.yml down -v
	rm -rf edge-agent/__pycache__ cloud-platform/__pycache__

# Development
shell-cloud:
	docker-compose -f infra/docker-compose.yml exec cloud /bin/sh

shell-edge:
	docker-compose -f infra/docker-compose.yml exec edge /bin/sh

shell-db:
	docker-compose -f infra/docker-compose.yml exec postgres psql -U onahealth -d onahealth
