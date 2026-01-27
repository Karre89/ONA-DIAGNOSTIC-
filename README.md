# Ona Health - Edge-First Medical Imaging Platform

**"See Clearly. Act Quickly."**

Edge-first AI diagnostics for TB triage in Africa. Works offline, syncs outputs only.

## Architecture

```
FACILITY (Ona Edge):
  X-ray machine → DICOM → Edge Agent → Local UI

CLOUD (Ona Cloud):
  Tenant management, device registry, reporting
```

## Quick Start

```bash
# Build and start all services
docker-compose -f infra/docker-compose.yml up --build

# Seed test data (tenant, site, device)
make seed

# Ingest a sample study
make ingest-sample

# View results
# Edge UI: http://localhost:8080
# Cloud API: curl http://localhost:8000/api/v1/results/recent
```

## Acceptance Tests

1. `docker-compose up` works
2. `make seed` creates tenant/site/device
3. `make ingest-sample` triggers processing
4. `localhost:8080` shows local UI with result
5. Cloud `/api/v1/results/recent` returns the record
6. Stop cloud → edge still works → restart cloud → backlog syncs

## Components

- **edge-agent**: Local inference, de-ID, UI, sync client
- **cloud-platform**: Tenant management, device registry, reporting
- **infra**: Docker compose, environment configs

## Tech Stack

- Python 3.11+, FastAPI
- Postgres (cloud), SQLite (edge)
- Docker + docker-compose
- JWT users, device token edge

## Non-Negotiables

- Outputs only to cloud (no images)
- De-ID on edge
- Append-only results
- NOT_CONFIDENT bucket for low quality
