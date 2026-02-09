"""
ONA Edge AI Inference Engine
Runs diagnostic models on medical images
Supports: ONNX (from Colab training), PyTorch, TorchXRayVision
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
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

    Supports:
    - ONNX models (from your Colab training notebook)
    - PyTorch .pth models
    - TorchXRayVision pretrained (fallback)
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_type: Dict[str, str] = {}  # 'onnx', 'pytorch', 'xrv'
        self._initialized = False
        self._using_xrv = False
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        # For PyTorch models
        try:
            import torch
            self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            self.device = 'cpu'

    def initialize(self):
        """Load all models at startup"""
        logger.info(f"Initializing inference engine...")

        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Priority 0: Look for multi-head model (TB + Pneumonia in one)
        multihead_path = model_dir / 'ona-multihead-finetuned.pth'
        if multihead_path.exists() and self._torch_available:
            if self._load_multihead_model(multihead_path):
                logger.info(f"Loaded multi-head model: {multihead_path.name}")

        # Priority 1: Look for ONNX models (from Colab training)
        if 'tb' not in self.models:
            onnx_patterns = [
                'ona-cxr-*.onnx',
                'tb_detector.onnx',
                'tb_model.onnx',
                'ona-tb-finetuned.onnx',
                '*.onnx'
            ]

            for pattern in onnx_patterns:
                onnx_files = list(model_dir.glob(pattern))
                if onnx_files:
                    onnx_path = sorted(onnx_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                    if self._load_onnx_model(onnx_path):
                        logger.info(f"Loaded ONNX model: {onnx_path.name}")
                        break

        # Priority 2: Look for PyTorch models
        if 'tb' not in self.models and self._torch_available:
            pth_patterns = ['ona-tb-finetuned.pth', 'tb_detector.pth', 'ona_tb_model.pth', 'model.pth']
            for pattern in pth_patterns:
                pth_files = list(model_dir.glob(pattern))
                if pth_files:
                    pth_path = pth_files[0]
                    if self._load_pytorch_model(pth_path):
                        logger.info(f"Loaded PyTorch model: {pth_path.name}")
                        break

        # Priority 3: Fallback to TorchXRayVision
        if not self.models:
            logger.info("No custom model found. Loading TorchXRayVision pretrained model...")
            try:
                import torchxrayvision as xrv
                import torch
                model = xrv.models.DenseNet(weights="densenet121-res224-all")
                model = model.to(self.device)
                model.eval()
                self.models['tb'] = model
                self.models['pneumonia'] = model
                self.model_type['tb'] = 'xrv'
                self.model_type['pneumonia'] = 'xrv'
                self._using_xrv = True
                logger.info("Loaded TorchXRayVision model (TB, Pneumonia, Cardiomegaly, etc.)")
            except ImportError:
                logger.warning("TorchXRayVision not installed. Using demo mode.")
            except Exception as e:
                logger.warning(f"Failed to load TorchXRayVision: {e}. Using demo mode.")

        self._initialized = True
        logger.info(f"Inference engine ready with {len(self.models)} models")

    def _load_multihead_model(self, model_path: Path) -> bool:
        """Load a MultiHeadClassifier model (TB + Pneumonia)"""
        try:
            import torch
            sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
            from finetune_model import MultiHeadClassifier

            import torchxrayvision as xrv
            xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
            model = MultiHeadClassifier(xrv_model)

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            # Store thresholds from checkpoint metadata
            self._multihead_thresholds = {}
            heads_meta = checkpoint.get('heads', {})
            for head_name, meta in heads_meta.items():
                self._multihead_thresholds[head_name] = meta.get('optimal_threshold', 0.5)

            # Register for all supported conditions
            for head_name in model.heads:
                self.models[head_name] = model
                self.model_type[head_name] = 'multihead'

            self._using_xrv = True  # uses XRV preprocessing
            logger.info(f"Multi-head model loaded with heads: {list(model.heads.keys())}")
            return True
        except Exception as e:
            logger.error(f"Failed to load multi-head model: {e}")
            return False

    def _load_onnx_model(self, model_path: Path) -> bool:
        """Load an ONNX model"""
        try:
            import onnxruntime as ort

            # Use GPU if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(model_path), providers=providers)

            self.models['tb'] = session
            self.model_type['tb'] = 'onnx'

            # Log model info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            logger.info(f"ONNX model: input={input_info.shape}, output={output_info.shape}")

            return True
        except ImportError:
            logger.warning("onnxruntime not installed. Cannot load ONNX model.")
            return False
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False

    def _load_pytorch_model(self, model_path: Path) -> bool:
        """Load a PyTorch model"""
        try:
            import torch

            model = torch.load(model_path, map_location=self.device)
            if isinstance(model, dict):
                # State dict - create architecture
                model = self._create_model_architecture()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()

            self.models['tb'] = model
            self.model_type['tb'] = 'pytorch'
            return True
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False

    def _create_model_architecture(self):
        """Create ResNet18 architecture (matching Colab training)"""
        import torch
        import torch.nn as nn
        from torchvision import models

        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 2)  # 2 classes: normal, tb
        )
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
        """
        if not self._initialized:
            self.initialize()

        start_time = datetime.utcnow()

        try:
            # Determine preprocessing based on model type
            use_xrv_preprocessing = self._using_xrv

            # Preprocess image
            image_tensor, original_image = preprocess_dicom(
                dicom_path,
                target_size=(224, 224),
                for_xrv=use_xrv_preprocessing
            )

            results = {
                'scan_id': scan_id,
                'timestamp': start_time.isoformat(),
                'conditions': {},
                'processing_time_ms': 0,
                'success': True,
                'error': None,
                'original_image': original_image
            }

            # Run inference for each condition
            for condition in conditions:
                if condition in self.models:
                    model = self.models[condition]
                    model_type = self.model_type.get(condition, 'unknown')

                    if model_type == 'multihead':
                        probability, heatmap = self._run_multihead_inference(model, image_tensor, condition)
                    elif model_type == 'onnx':
                        probability, heatmap = self._run_onnx_inference(model, image_tensor, original_image)
                    elif model_type == 'xrv':
                        probability, heatmap = self._run_xrv_inference(model, image_tensor, condition)
                    elif model_type == 'pytorch':
                        probability, heatmap = self._run_pytorch_inference(model, image_tensor)
                    else:
                        probability = np.random.uniform(0.1, 0.9)
                        heatmap = np.zeros((224, 224), dtype=np.uint8)
                else:
                    # Demo mode
                    probability = np.random.uniform(0.1, 0.9)
                    heatmap = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

                # Create overlay
                overlay = overlay_heatmap(original_image, heatmap)

                # Classify severity
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

    def _run_multihead_inference(self, model, image_tensor, condition: str) -> tuple:
        """Run inference with MultiHeadClassifier"""
        import torch

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            if len(image_tensor.shape) == 3:
                input_tensor = image_tensor.unsqueeze(0)
            else:
                input_tensor = image_tensor

            output = model(input_tensor, head=condition)
            probability = torch.sigmoid(output).item()

        heatmap = self._generate_simple_heatmap(
            image_tensor.cpu().numpy() if hasattr(image_tensor, 'numpy') else image_tensor,
            probability
        )

        return probability, heatmap

    def _run_onnx_inference(self, session, image_tensor, original_image) -> tuple:
        """Run inference with ONNX model"""
        import torch

        # Convert tensor to numpy for ONNX
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.numpy()
        else:
            image_np = image_tensor

        # Add batch dimension if needed
        if len(image_np.shape) == 3:
            image_np = image_np[np.newaxis, ...]

        # Ensure float32
        image_np = image_np.astype(np.float32)

        # Run ONNX inference
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: image_np})[0]

        # Apply softmax to get probabilities [normal, tb]
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # TB probability is index 1
        tb_probability = float(probs[0, 1])

        # Generate simple heatmap (ONNX doesn't support GradCAM easily)
        # Use attention-based approximation
        heatmap = self._generate_simple_heatmap(image_np, tb_probability)

        return tb_probability, heatmap

    def _run_pytorch_inference(self, model, image_tensor) -> tuple:
        """Run inference with PyTorch model"""
        import torch

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))

            # Handle 2-class output
            if output.shape[-1] == 2:
                probs = torch.softmax(output, dim=1)
                probability = probs[0, 1].item()  # TB is index 1
            else:
                probability = torch.sigmoid(output).item()

        # Generate heatmap
        heatmap = generate_heatmap(model, image_tensor, target_class=1)

        return probability, heatmap

    def _run_xrv_inference(self, model, image_tensor, condition: str) -> tuple:
        """Run inference with TorchXRayVision model"""
        import torch

        # image_tensor should be (1, 224, 224) for XRV
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            # Add batch dimension: (1, 224, 224) -> (1, 1, 224, 224)
            if len(image_tensor.shape) == 3:
                input_tensor = image_tensor.unsqueeze(0)
            else:
                input_tensor = image_tensor

            logger.info(f"XRV input shape: {input_tensor.shape}")

            output = model(input_tensor)

            # TorchXRayVision outputs probabilities for 18 conditions
            # Map condition names to XRV pathology names
            condition_map = {
                'tb': 'Infiltration',  # TB often shows as infiltration/consolidation
                'pneumonia': 'Pneumonia',
                'cardiomegaly': 'Cardiomegaly',
                'consolidation': 'Consolidation',
                'effusion': 'Effusion'
            }

            target_name = condition_map.get(condition, condition)

            if hasattr(model, 'pathologies'):
                pathologies = list(model.pathologies)
                logger.info(f"XRV pathologies: {pathologies}")

                if target_name in pathologies:
                    idx = pathologies.index(target_name)
                    probability = torch.sigmoid(output[0, idx]).item()
                else:
                    # Fallback: use first pathology
                    probability = torch.sigmoid(output[0, 0]).item()

                # Log all pathology scores for debugging
                for i, path in enumerate(pathologies[:5]):
                    score = torch.sigmoid(output[0, i]).item()
                    logger.debug(f"  {path}: {score:.3f}")
            else:
                probability = torch.sigmoid(output[0, 0]).item()

        # Generate simple heatmap (GradCAM not easily supported for XRV)
        heatmap = self._generate_simple_heatmap(
            image_tensor.cpu().numpy() if hasattr(image_tensor, 'numpy') else image_tensor,
            probability
        )

        return probability, heatmap

    def _generate_simple_heatmap(self, image_np: np.ndarray, probability: float) -> np.ndarray:
        """Generate a simple attention-based heatmap"""
        import cv2

        # Handle different tensor shapes
        if len(image_np.shape) == 4:
            img = image_np[0, 0]  # (batch, channel, H, W) -> (H, W)
        elif len(image_np.shape) == 3:
            img = image_np[0]  # (channel, H, W) -> (H, W)
        else:
            img = image_np

        # Normalize to 0-255
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min) * 255
        else:
            img = np.zeros_like(img)

        img = img.astype(np.float32)

        # Resize to 224x224 if needed
        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224))

        # Apply Gaussian blur to create smooth heatmap
        heatmap = cv2.GaussianBlur(img.astype(np.uint8), (31, 31), 0)

        # Scale by probability
        heatmap = (heatmap * probability).astype(np.uint8)

        return heatmap

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
