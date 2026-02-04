# ONA Edge - AI-Powered Medical Diagnostics

Edge software for ONA Health's AI diagnostic platform. Runs on clinic hardware (NVIDIA Jetson or Intel NUC) to provide offline-capable TB screening from X-rays.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ONA EDGE                                │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  DICOM   │──▶│    AI    │──▶│  Local   │   │   Sync   │ │
│  │ Listener │   │ Inference│   │    UI    │   │ Service  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       ▲                                              │      │
└───────│──────────────────────────────────────────────│──────┘
        │                                              │
        │ DICOM                                        │ HTTPS
        │                                              ▼
┌───────────────┐                              ┌──────────────┐
│  X-RAY MACHINE │                              │  ONA CLOUD   │
└───────────────┘                              └──────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your device settings
```

### 3. Add AI Model

Place your trained model in `data/models/`:
- `tb_detector.pth` - TB detection model

### 4. Run

```bash
python main.py
```

## Docker Deployment

### Build and Run

```bash
docker-compose up -d
```

### For NVIDIA Jetson

Uncomment the GPU section in `docker-compose.yml` and use:

```bash
docker-compose up -d
```

## Ports

- **11112** - DICOM listener (receives X-rays)
- **8080** - Web UI (nurse interface)

## Testing

### Test DICOM Connectivity

```bash
# Install pynetdicom
pip install pynetdicom

# Send echo (connectivity test)
python -m pynetdicom echoscu localhost 11112 -aet TESTER -aec ONA_EDGE

# Send test DICOM file
python -m pynetdicom storescu localhost 11112 test.dcm -aet TESTER -aec ONA_EDGE
```

### Access Web UI

Open http://localhost:8080 in your browser.

## Project Structure

```
ona-edge/
├── main.py              # Application entry point
├── config.py            # Configuration
├── dicom/               # DICOM listener module
│   ├── listener.py      # DICOM SCP server
│   └── handlers.py      # Image handling
├── inference/           # AI inference module
│   ├── engine.py        # Inference orchestrator
│   ├── preprocessing.py # Image preprocessing
│   └── postprocessing.py# Heatmap generation
├── ui/                  # Local web UI
│   ├── server.py        # FastAPI server
│   └── templates/       # HTML templates
├── sync/                # Cloud sync module
│   └── service.py       # Sync service
├── data/                # Data directory
│   ├── models/          # AI models
│   ├── scans/           # Received scans
│   └── logs/            # Log files
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| ONA_DEVICE_ID | ONA-DEV-001 | Unique device identifier |
| ONA_SITE_ID | site-001 | Site/clinic identifier |
| ONA_CLOUD_API | https://... | ONA Cloud API URL |
| ONA_API_KEY | | Device authentication token |
| ONA_DICOM_PORT | 11112 | DICOM listener port |
| ONA_UI_PORT | 8080 | Web UI port |
| ONA_DEVICE | cpu | AI device (cpu/cuda) |

## Clinic Installation

1. Connect edge device to clinic network
2. Configure X-ray machine to send DICOM to edge device IP:11112
3. Access UI at http://[device-ip]:8080
4. Verify with test scan

## Support

- Logs: `data/logs/ona-edge.log`
- Status: http://localhost:8080/api/status
