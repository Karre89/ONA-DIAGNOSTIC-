# ONA Health - Model Training Guide

## Overview

The ONA platform uses a **federated learning** approach where:
1. Data is collected from edge devices
2. Training happens in the cloud
3. Updated models are pushed back to edges

---

## Current State (v1.0 - Stub Model)

The current implementation uses a **mock inference** that generates random scores for demonstration purposes.

```python
# Current stub in inference.py
tb_score = random.uniform(0, 1)
quality_score = random.uniform(0.6, 1.0)
```

---

## Production Training Pipeline

### Phase 1: Data Collection

```
┌──────────────────────────────────────────────────────────────┐
│                     EDGE DEVICES                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Kenya   │  │ Uganda  │  │ Tanzania│  │ Nigeria │         │
│  │ 500/day │  │ 300/day │  │ 400/day │  │ 600/day │         │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘         │
│       │            │            │            │               │
│       ▼            ▼            ▼            ▼               │
│  ┌─────────────────────────────────────────────────┐        │
│  │              CLOUD DATABASE                      │        │
│  │                                                  │        │
│  │  Data Collected:                                │        │
│  │  - De-identified DICOM images                   │        │
│  │  - AI predictions (score, risk bucket)          │        │
│  │  - Clinician feedback (Agree/Disagree/Unsure)   │        │
│  │  - Ground truth (GeneXpert results)             │        │
│  │  - Patient outcomes (treatment success)         │        │
│  │                                                  │        │
│  └─────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

### Phase 2: Data Labeling

| Label Source | Reliability | Use Case |
|-------------|-------------|----------|
| **GeneXpert Result** | Gold standard | Primary training label |
| **Clinician Agree** | High | Confirms AI was correct |
| **Clinician Disagree** | High | Corrects AI mistakes |
| **Sputum Culture** | Gold standard | Backup confirmation |
| **Treatment Outcome** | Medium | Long-term validation |

### Phase 3: Model Architecture

```python
# Typical TB detection model architecture
import torch
import torch.nn as nn
from torchvision import models

class TBDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pre-trained DenseNet as backbone
        self.backbone = models.densenet121(pretrained=True)

        # Replace classifier for TB detection
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # [tb_score, quality_score, abnormal_score]
        )

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))
```

### Phase 4: Training Process

```python
# Training loop (simplified)
def train_model(model, train_loader, val_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_metrics = evaluate(model, val_loader)

        print(f"Epoch {epoch}: Loss={loss:.4f}, AUC={val_metrics['auc']:.4f}")

    return model
```

### Phase 5: Model Validation

| Metric | Target | Description |
|--------|--------|-------------|
| **Sensitivity** | >95% | Catch most TB cases (minimize false negatives) |
| **Specificity** | >80% | Minimize false positives |
| **AUC-ROC** | >0.90 | Overall discriminative ability |
| **NPV** | >99% | Negative predictive value |

### Phase 6: Model Export

```python
# Export to ONNX for edge deployment
import torch.onnx

def export_model(model, version):
    model.eval()
    dummy_input = torch.randn(1, 1, 512, 512)  # Grayscale X-ray

    torch.onnx.export(
        model,
        dummy_input,
        f"ona-cxr-tb-{version}.onnx",
        input_names=['image'],
        output_names=['scores'],
        dynamic_axes={'image': {0: 'batch'}}
    )
```

### Phase 7: Edge Deployment

```
┌──────────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ona-cxr-tb-v1.0.onnx  │  Released: 2026-01-01      │   │
│  │  ona-cxr-tb-v1.1.onnx  │  Released: 2026-02-01      │   │
│  │  ona-cxr-tb-v1.2.onnx  │  Released: 2026-03-01  ←   │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                  │
│                            ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              EDGE UPDATE PROCESS                      │   │
│  │                                                       │   │
│  │  1. Edge checks for updates (hourly)                 │   │
│  │  2. Downloads new model if available                 │   │
│  │  3. Validates model signature                        │   │
│  │  4. Runs sanity checks on test images               │   │
│  │  5. Hot-swaps to new model                          │   │
│  │  6. Reports success to cloud                        │   │
│  │                                                       │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Implementing Real Inference

To upgrade from stub to real model:

### Step 1: Add ONNX Runtime

```toml
# edge-agent/pyproject.toml
dependencies = [
    # ... existing deps
    "onnxruntime>=1.16.0",
]
```

### Step 2: Update Inference Service

```python
# edge-agent/app/inference/inference.py

import onnxruntime as ort
import numpy as np
from PIL import Image

class InferenceService:
    def __init__(self, db: Session):
        self.db = db
        self.model_path = os.path.join(settings.model_dir, "ona-cxr-tb-v1.0.onnx")
        self.session = None
        self._load_model()

    def _load_model(self):
        """Load ONNX model"""
        if os.path.exists(self.model_path):
            self.session = ort.InferenceSession(self.model_path)
            logger.info(f"Loaded model: {self.model_path}")
        else:
            logger.warning("No model found, using stub inference")

    def _preprocess(self, image_path: str) -> np.ndarray:
        """Preprocess image for model input"""
        # Load and resize
        img = Image.open(image_path).convert('L')
        img = img.resize((512, 512))

        # Normalize to [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0

        # Add batch and channel dimensions
        arr = arr[np.newaxis, np.newaxis, :, :]

        return arr

    def _run_inference(self, image_path: str) -> dict:
        """Run model inference"""
        if self.session is None:
            # Fall back to stub
            return self._stub_inference()

        # Preprocess
        input_data = self._preprocess(image_path)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data})

        # Parse outputs
        scores = outputs[0][0]
        return {
            "tb_score": float(scores[0]),
            "quality_score": float(scores[1]),
            "abnormal_score": float(scores[2])
        }
```

### Step 3: Model Update Daemon

```python
# edge-agent/app/device_daemon/daemon.py

async def check_model_updates(self):
    """Check for and download new model versions"""
    try:
        response = await self.client.get(
            f"{self.cloud_url}/api/v1/models/manifest",
            headers=self.get_auth_headers()
        )

        if response.status_code == 200:
            manifest = response.json()
            latest = manifest.get("latest_version")

            if latest != self.current_version:
                logger.info(f"New model available: {latest}")
                await self._download_model(latest)

    except Exception as e:
        logger.error(f"Model update check failed: {e}")

async def _download_model(self, version: str):
    """Download and install new model"""
    url = f"{self.cloud_url}/api/v1/models/{version}/download"

    response = await self.client.get(url, headers=self.get_auth_headers())

    if response.status_code == 200:
        model_path = os.path.join(settings.model_dir, f"ona-cxr-tb-{version}.onnx")

        with open(model_path, "wb") as f:
            f.write(response.content)

        # Validate model
        if self._validate_model(model_path):
            # Update symlink
            current_path = os.path.join(settings.model_dir, "ona-cxr-tb-current.onnx")
            os.symlink(model_path, current_path)

            logger.info(f"Model updated to {version}")
```

---

## Active Learning

The system can request human review for uncertain cases:

```python
def get_risk_bucket(tb_score: float, quality_score: float) -> str:
    # Request review for uncertain predictions
    if 0.4 <= tb_score <= 0.6:
        return "NEEDS_REVIEW"  # AI is uncertain

    if quality_score < 0.7:
        return "NOT_CONFIDENT"
    if tb_score >= 0.6:
        return "HIGH"
    if tb_score >= 0.3:
        return "MEDIUM"
    return "LOW"
```

These "NEEDS_REVIEW" cases are prioritized for clinician feedback, which becomes high-value training data.

---

## Privacy & Security

| Requirement | Implementation |
|-------------|----------------|
| **De-identification** | All PHI stripped before cloud upload |
| **Encryption** | TLS 1.3 for transit, AES-256 for storage |
| **Access Control** | Role-based access to training data |
| **Audit Trail** | All data access logged |
| **Data Retention** | Configurable per-tenant policies |

---

## Training Schedule

| Frequency | Activity |
|-----------|----------|
| **Daily** | Collect new labeled data |
| **Weekly** | Retrain model with new data |
| **Monthly** | Full validation on held-out test set |
| **Quarterly** | External audit of model performance |

---

## Metrics Dashboard (Cloud)

The cloud platform would track:

- Model accuracy over time
- Performance by site/region
- Clinician agreement rates
- False positive/negative trends
- Data quality metrics

---

## Next Steps to Implement

1. **Acquire training data** - Partner with hospitals that have labeled X-rays
2. **Set up training infrastructure** - GPU servers for model training
3. **Implement model registry** - Version control for models
4. **Add OTA updates** - Automatic model deployment to edges
5. **Build monitoring dashboard** - Track model performance in production
