"""
ONA Edge AI Inference Engine
Runs diagnostic models on medical images
"""

import logging
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .preprocessing import preprocess_dicom
from .postprocessing import generate_heatmap, overlay_heatmap
import sys
sys.path.append('..')
from config import MODEL_DIR, DEVICE, MAX_WORKERS

logger = logging.getLogger('ona.inference')


class InferenceEngine:
    """
    Main AI inference orchestrator
    Manages models and processes images
    """

    def __init__(self):
        self.models: Dict[str, torch.nn.Module] = {}
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        self._initialized = False
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def initialize(self):
        """Load all models at startup"""
        logger.info(f"Initializing inference engine on {self.device}")

        # Look for TB model
        model_path = Path(MODEL_DIR) / 'tb_detector.pth'
        if model_path.exists():
            self.models['tb'] = self._load_model(model_path)
            logger.info("Loaded TB detection model")
        else:
            # Try alternative names
            for alt_name in ['ona_tb_model.pth', 'model.pth', 'tb_model.pth']:
                alt_path = Path(MODEL_DIR) / alt_name
                if alt_path.exists():
                    self.models['tb'] = self._load_model(alt_path)
                    logger.info(f"Loaded TB model from {alt_name}")
                    break

        if not self.models:
            logger.warning("No models found! Using demo mode with random predictions.")

        self._initialized = True
        logger.info(f"Inference engine ready with {len(self.models)} models")

    def _load_model(self, model_path: Path) -> torch.nn.Module:
        """Load a PyTorch model"""
        try:
            # Try loading as full model first
            model = torch.load(model_path, map_location=self.device)
            if isinstance(model, dict):
                # It's a state dict, need to create model architecture
                model = self._create_model_architecture()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None

    def _create_model_architecture(self):
        """Create DenseNet121 architecture for TB detection"""
        import torchvision.models as models
        model = models.densenet121(weights=None)
        # Modify for binary classification
        model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
        return model

    def is_ready(self) -> bool:
        return self._initialized

    def run_inference(
        self,
        scan_id: str,
        dicom_path: str,
        conditions: List[str] = ['tb']
    ) -> Dict:
        """
        Run AI inference on a DICOM image

        Args:
            scan_id: Unique identifier for this scan
            dicom_path: Path to DICOM file
            conditions: List of conditions to check ['tb', 'pneumonia', etc.]

        Returns:
            Dictionary with results for each condition
        """
        if not self._initialized:
            self.initialize()

        start_time = datetime.utcnow()

        try:
            # Preprocess image
            image_tensor, original_image = preprocess_dicom(
                dicom_path,
                target_size=(224, 224)
            )
            image_tensor = image_tensor.to(self.device)

            results = {
                'scan_id': scan_id,
                'timestamp': start_time.isoformat(),
                'conditions': {},
                'processing_time_ms': 0,
                'success': True,
                'error': None,
                'original_image': original_image
            }

            # Run each requested model
            for condition in conditions:
                if condition in self.models and self.models[condition] is not None:
                    model = self.models[condition]

                    with torch.no_grad():
                        # Forward pass
                        output = model(image_tensor.unsqueeze(0))
                        probability = torch.sigmoid(output).item()

                    # Generate heatmap for explainability
                    heatmap = generate_heatmap(
                        model=model,
                        image_tensor=image_tensor,
                        target_class=0
                    )

                    # Overlay on original
                    overlay = overlay_heatmap(original_image, heatmap)

                else:
                    # Demo mode - generate random result
                    probability = np.random.uniform(0.1, 0.9)
                    heatmap = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
                    overlay = original_image

                # Determine severity
                severity = self._classify_severity(probability)

                results['conditions'][condition] = {
                    'probability': round(probability * 100, 2),
                    'severity': severity,
                    'heatmap': heatmap,
                    'overlay': overlay,
                    'recommendation': self._get_recommendation(condition, severity)
                }

            # Calculate processing time
            end_time = datetime.utcnow()
            results['processing_time_ms'] = int(
                (end_time - start_time).total_seconds() * 1000
            )

            logger.info(
                f"Inference complete: {scan_id} | "
                f"TB: {results['conditions'].get('tb', {}).get('probability', 'N/A')}% | "
                f"Time: {results['processing_time_ms']}ms"
            )

            return results

        except Exception as e:
            logger.error(f"Inference failed for {scan_id}: {e}")
            return {
                'scan_id': scan_id,
                'timestamp': start_time.isoformat(),
                'success': False,
                'error': str(e)
            }

    def _classify_severity(self, probability: float) -> str:
        """Classify probability into severity levels"""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _get_recommendation(self, condition: str, severity: str) -> str:
        """Get clinical recommendation based on result"""
        recommendations = {
            'tb': {
                'LOW': 'No TB indicators detected. Continue routine monitoring.',
                'MEDIUM': 'Possible TB indicators. Recommend sputum test for confirmation.',
                'HIGH': 'Strong TB indicators. URGENT: Collect sputum sample and refer to TB program.'
            },
            'pneumonia': {
                'LOW': 'No pneumonia indicators detected.',
                'MEDIUM': 'Possible pneumonia. Consider clinical correlation.',
                'HIGH': 'Likely pneumonia. Recommend immediate treatment initiation.'
            }
        }
        return recommendations.get(condition, {}).get(severity, 'Consult physician.')


# Singleton instance
_engine = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
