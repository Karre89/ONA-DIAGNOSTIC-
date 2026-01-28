# ONA Health - Edge-First Medical Imaging Platform

**"See Clearly. Act Quickly."**

Edge-first AI diagnostics for TB triage in Africa. Works offline, syncs outputs only.

## Architecture

```
FACILITY (Edge Device):
  X-ray Machine → DICOM SCP (port 104) → Edge Agent → AI Inference → Local UI (port 8080)
                                                ↓
                                         Sync to Cloud (outputs only)

CLOUD (Ona Cloud - port 8000):
  Tenant management, device registry, results aggregation, reporting
```

## Quick Start

### 1. Start the Platform
```bash
cd infra
docker-compose up --build
```

Wait for all services to show "Application startup complete"

### 2. Access the UI
- **Edge UI**: http://localhost:8080
- **Cloud API**: http://localhost:8000/api/v1

### 3. Test with Sample Image
```bash
# PowerShell
Invoke-WebRequest -Uri http://localhost:8080/api/ingest-sample -Method POST

# Or curl
curl -X POST http://localhost:8080/api/ingest-sample
```

## How to Ingest Images

### Option 1: DICOM SCP (Production)
Connect X-ray equipment to send DICOM images directly:
- **Port**: 104
- **AE Title**: ONA_EDGE
- Images are automatically processed when received

### Option 2: API Upload
```bash
# Upload a DICOM file
curl -X POST http://localhost:8080/api/ingest \
  -F "file=@/path/to/xray.dcm"

# Upload a PNG/JPG file
curl -X POST http://localhost:8080/api/ingest \
  -F "file=@/path/to/xray.png"
```

### Option 3: Test Sample
```bash
curl -X POST http://localhost:8080/api/ingest-sample
```

## Project Structure

```
imaging-platform/
├── cloud-platform/          # Cloud backend (FastAPI)
│   ├── app/
│   │   ├── api/             # REST endpoints
│   │   ├── core/            # Config, database
│   │   └── models/          # SQLAlchemy models
│   └── Dockerfile
│
├── edge-agent/              # Edge device software (FastAPI)
│   ├── app/
│   │   ├── dicom_ingest/    # DICOM SCP receiver
│   │   ├── deid/            # De-identification
│   │   ├── inference/       # AI model inference
│   │   ├── sync_client/     # Cloud sync with retry
│   │   ├── local_ui/        # Web UI (Jinja2 templates)
│   │   └── device_daemon/   # Heartbeat, model updates
│   ├── data/
│   │   └── models/          # ONNX model files
│   └── Dockerfile
│
├── ml/                      # Machine Learning
│   └── ONA_Train_TB_Model.ipynb  # Train & export model (Colab)
│
├── infra/                   # Infrastructure
│   └── docker-compose.yml   # Local development stack
│
└── README.md
```

## AI Model

### Current Model: ResNet18 v2.0
- **Architecture**: ResNet18 (pretrained on ImageNet, fine-tuned on TB data)
- **Input**: 224x224 RGB image
- **Output**: TB probability (0-100%)
- **Risk Buckets**:
  - LOW: < 30% TB probability
  - MEDIUM: 30-60% TB probability
  - HIGH: > 60% TB probability
  - NOT_CONFIDENT: Image quality < 70%

### Retraining the Model
1. Open `ml/ONA_Train_TB_Model.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Click Runtime → Run all
4. Download the `.onnx` file (~45MB single file)
5. Copy to `edge-agent/data/models/`
6. Restart the edge container

## API Endpoints

### Edge Agent (localhost:8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/studies` | GET | All studies list |
| `/result/{id}` | GET | Result detail page |
| `/api/ingest` | POST | Upload image for processing |
| `/api/ingest-sample` | POST | Process test sample |
| `/api/model-info` | GET | Current model information |
| `/health` | GET | Health check |

### Cloud Platform (localhost:8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/devices/register` | POST | Register new device |
| `/api/v1/devices/{id}/heartbeat` | POST | Device heartbeat |
| `/api/v1/results` | POST | Upload result from edge |
| `/api/v1/results/recent` | GET | Recent results |
| `/api/v1/seed` | POST | Create test tenant/site/device |
| `/health` | GET | Health check |

## Security Features

- **De-identification**: Patient data stripped on edge before cloud sync
- **Outputs Only**: Only results sync to cloud, never raw images
- **Device Authentication**: Token-based device registration
- **Local Processing**: AI runs locally, works offline

## Offline Operation

The edge agent works fully offline:
1. Receives and processes images locally
2. Stores results in local SQLite database
3. Queues results for sync when cloud is unavailable
4. Automatically syncs backlog when connection restored

## Languages Supported

- English (en)
- Swahili (sw)
- Somali (so)

Change language: Add `?lang=sw` to any URL

## Development

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- Google Colab account (for model training)

### Local Development
```bash
# Start services
cd infra
docker-compose up --build

# View logs
docker-compose logs -f edge

# Rebuild after code changes
docker-compose up --build edge
```

### Running Tests
```bash
# Edge agent tests
cd edge-agent
pytest

# Cloud platform tests
cd cloud-platform
pytest
```

## Troubleshooting

### Model shows "stub"
- Check model files exist in `edge-agent/data/models/`
- Should have both `.onnx` and `.onnx.data` files (or single combined `.onnx`)
- Restart edge container after adding model

### Sync not working
- Check cloud container is running
- Check heartbeat messages in logs
- Results queue locally and sync when cloud available

### DICOM not receiving
- Check port 104 is exposed
- Verify AE Title matches: ONA_EDGE
- Check DICOM sender configuration

## License

Proprietary - ONA Health
