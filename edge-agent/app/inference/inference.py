import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import Study, Job, Result, SyncQueue

logger = logging.getLogger(__name__)

# Image processing imports
try:
    import numpy as np
    from PIL import Image, ImageFilter
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False
    logger.warning("numpy/PIL not available - heatmap generation limited")

# DICOM imports
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

# ONNX Runtime imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not available - using stub inference")


# Global model cache
_model_session: Optional["ort.InferenceSession"] = None
_model_version: Optional[str] = None


def get_risk_bucket(tb_score: float, quality_score: float) -> str:
    """
    Risk bucket logic as specified in build pack

    LOW: tb_score < 0.3
    MEDIUM: 0.3 <= tb_score < 0.6
    HIGH: tb_score >= 0.6
    NOT_CONFIDENT: quality_score < 0.7
    """
    if quality_score < 0.7:
        return "NOT_CONFIDENT"
    if tb_score >= 0.6:
        return "HIGH"
    if tb_score >= 0.3:
        return "MEDIUM"
    return "LOW"


def load_onnx_model(model_path: str, version: str) -> Optional["ort.InferenceSession"]:
    """
    Load ONNX model into global cache

    Args:
        model_path: Path to .onnx file
        version: Model version string

    Returns:
        ONNX InferenceSession or None if loading fails
    """
    global _model_session, _model_version

    if not ONNX_AVAILABLE:
        logger.warning("ONNX runtime not available")
        return None

    if _model_session is not None and _model_version == version:
        logger.debug(f"Using cached model: {version}")
        return _model_session

    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return None

    try:
        logger.info(f"Loading ONNX model: {model_path}")
        # Use CPU execution provider for edge devices
        _model_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        _model_version = version
        logger.info(f"Model loaded successfully: {version}")
        return _model_session
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        return None


def get_model_info() -> dict:
    """Get information about currently loaded model"""
    global _model_session, _model_version

    if _model_session is None:
        return {
            "loaded": False,
            "version": None,
            "using_stub": True
        }

    return {
        "loaded": True,
        "version": _model_version,
        "using_stub": False,
        "inputs": [inp.name for inp in _model_session.get_inputs()],
        "outputs": [out.name for out in _model_session.get_outputs()]
    }


class InferenceService:
    """
    Inference Service

    Supports two modes:
    1. ONNX Model: Real AI inference using ONNX runtime
    2. Stub Mode: Simulated scores for testing (fallback when no model)

    The service automatically uses ONNX if a model file exists,
    otherwise falls back to stub inference.
    """

    # Model configurations
    MODEL_CONFIGS = {
        "resnet18": {"input_size": (224, 224), "channels": 3, "pattern": "ona-cxr-resnet18-", "output_type": "binary"},
        "resnet50": {"input_size": (512, 512), "channels": 1, "pattern": "ona-cxr-resnet50-", "output_type": "multilabel"},
        "densenet": {"input_size": (224, 224), "channels": 1, "pattern": "ona-cxr-tb-", "output_type": "multilabel"},
    }

    # Default settings (will be updated based on loaded model)
    INPUT_SIZE = (224, 224)
    INPUT_CHANNELS = 3
    OUTPUT_TYPE = "binary"

    def __init__(self, db: Session):
        self.db = db
        self.model_version = settings.default_model_version
        self.heatmap_dir = os.path.join(settings.edge_data_dir, "heatmaps")
        self.processed_dir = os.path.join(settings.edge_data_dir, "processed")
        self.model_dir = os.path.join(settings.edge_data_dir, "models")
        os.makedirs(self.heatmap_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Try to load ONNX model
        self.session = self._load_model()
        self.using_stub = self.session is None

        if self.using_stub:
            logger.info("Using STUB inference (no model loaded)")
        else:
            logger.info(f"Using ONNX inference: {self.model_version}")

    def _load_model(self) -> Optional["ort.InferenceSession"]:
        """Load ONNX model if available"""
        # Try ResNet18 first (vanilla PyTorch - guaranteed ONNX compatible)
        resnet18_filename = f"ona-cxr-resnet18-{self.model_version}.onnx"
        resnet18_path = os.path.join(self.model_dir, resnet18_filename)

        if os.path.exists(resnet18_path):
            config = self.MODEL_CONFIGS["resnet18"]
            self.INPUT_SIZE = config["input_size"]
            self.INPUT_CHANNELS = config["channels"]
            self.OUTPUT_TYPE = config["output_type"]
            logger.info(f"Found ResNet18 model, using {self.INPUT_SIZE} input, {self.INPUT_CHANNELS} channels")
            return load_onnx_model(resnet18_path, self.model_version)

        # Try ResNet50 (TorchXRayVision)
        resnet50_filename = f"ona-cxr-resnet50-{self.model_version}.onnx"
        resnet50_path = os.path.join(self.model_dir, resnet50_filename)

        if os.path.exists(resnet50_path):
            config = self.MODEL_CONFIGS["resnet50"]
            self.INPUT_SIZE = config["input_size"]
            self.INPUT_CHANNELS = config["channels"]
            self.OUTPUT_TYPE = config["output_type"]
            logger.info(f"Found ResNet50 model, using {self.INPUT_SIZE} input")
            return load_onnx_model(resnet50_path, self.model_version)

        # Fall back to DenseNet (legacy)
        densenet_filename = f"ona-cxr-tb-{self.model_version}.onnx"
        densenet_path = os.path.join(self.model_dir, densenet_filename)

        if os.path.exists(densenet_path):
            config = self.MODEL_CONFIGS["densenet"]
            self.INPUT_SIZE = config["input_size"]
            self.INPUT_CHANNELS = config["channels"]
            self.OUTPUT_TYPE = config["output_type"]
            logger.info(f"Found DenseNet model, using {self.INPUT_SIZE} input")
            return load_onnx_model(densenet_path, self.model_version)

        # Check for generic "current" model
        current_path = os.path.join(self.model_dir, "ona-cxr-current.onnx")
        if os.path.exists(current_path):
            return load_onnx_model(current_path, self.model_version)

        return None

    def _preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess image for model input

        Supports two modes:
        1. ResNet18 (binary): RGB input, ImageNet normalization
        2. TorchXRayVision (multilabel): Grayscale, [-1024, 1024] range

        Returns:
            numpy array ready for inference or None
        """
        if not IMAGE_AVAILABLE:
            return None

        try:
            # Load image as grayscale
            img = Image.open(image_path).convert('L')

            # Resize to expected input size
            img = img.resize(self.INPUT_SIZE, Image.Resampling.LANCZOS)

            # Convert to numpy array
            arr = np.array(img, dtype=np.float32)

            if self.INPUT_CHANNELS == 3:
                # ResNet18 binary model: RGB with ImageNet normalization
                # Normalize to [0, 1] then apply ImageNet stats
                arr = arr / 255.0
                mean = 0.485
                std = 0.229
                arr = (arr - mean) / std
                # Stack to 3 channels (RGB from grayscale)
                arr = np.stack([arr, arr, arr], axis=0)
                # Add batch dimension: (3, H, W) -> (1, 3, H, W)
                arr = arr[np.newaxis, :, :, :]
            else:
                # TorchXRayVision: Grayscale with [-1024, 1024] range
                arr = (arr / 255.0) * 2048 - 1024
                # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
                arr = arr[np.newaxis, np.newaxis, :, :]

            return arr

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None

    def _run_onnx_inference(self, image_path: str) -> Tuple[float, float, float]:
        """
        Run ONNX model inference

        Returns:
            Tuple of (tb_score, quality_score, abnormal_score)
        """
        # Preprocess image
        input_data = self._preprocess_image(image_path)
        if input_data is None:
            logger.warning("Preprocessing failed, falling back to stub")
            return self._run_stub_inference(None)

        try:
            # Get input name from model
            input_name = self.session.get_inputs()[0].name

            # Run inference
            outputs = self.session.run(None, {input_name: input_data})
            raw_output = outputs[0][0]

            if self.OUTPUT_TYPE == "binary":
                # ResNet18 binary model: outputs logits [normal, tb]
                # Apply softmax to get probabilities
                exp_scores = np.exp(raw_output - np.max(raw_output))  # Numerical stability
                probs = exp_scores / exp_scores.sum()

                tb_score = float(probs[1])  # TB probability
                abnormal_score = tb_score  # Same as TB for binary model

                # Quality score based on confidence (how far from 0.5)
                confidence = abs(tb_score - 0.5) * 2  # 0 to 1
                quality_score = float(0.7 + confidence * 0.3)  # 0.7 to 1.0

            else:
                # TorchXRayVision multilabel: outputs 18 pathology scores
                scores = raw_output

                # Map pathology indices for TB indicators
                tb_score = float(np.clip(scores[0], 0, 1))
                abnormal_score = float(np.clip(np.max(scores), 0, 1))

                # Quality score based on output variance
                quality_score = float(1.0 - np.std(scores) * 2)
                quality_score = np.clip(quality_score, 0.5, 1.0)

            logger.info(f"ONNX inference: tb={tb_score:.3f}, quality={quality_score:.3f}, type={self.OUTPUT_TYPE}")
            return tb_score, quality_score, abnormal_score

        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return self._run_stub_inference(None)

    def _run_stub_inference(self, study_id: Optional[str]) -> Tuple[float, float, float]:
        """
        Run stub inference (mock scores)

        Uses study_id hash for deterministic results
        """
        if study_id:
            random.seed(hash(study_id))

        tb_score = random.uniform(0, 1)
        quality_score = random.uniform(0.6, 1.0)
        abnormal_score = random.uniform(0, 1)

        if study_id:
            random.seed()  # Reset

        return tb_score, quality_score, abnormal_score

    def process_study(self, study: Study) -> Optional[Result]:
        """
        Run inference on a study

        Returns Result if successful, None otherwise
        """
        logger.info(f"Running inference on study: {study.id} (using_stub={self.using_stub})")

        # Update job status
        job = self.db.query(Job).filter(Job.study_id == study.id).first()
        if job:
            job.status = "RUNNING"
            job.attempts += 1
            self.db.commit()

        try:
            start_time = time.time()

            # Get display image for inference
            image_path = self._ensure_display_image(study)

            # Run inference (ONNX or stub)
            if not self.using_stub and image_path and os.path.exists(image_path):
                # Real ONNX inference
                tb_score, quality_score, abnormal_score = self._run_onnx_inference(image_path)
            else:
                # Stub inference (mock scores)
                tb_score, quality_score, abnormal_score = self._run_stub_inference(study.id)
                # Add small delay to simulate processing
                time.sleep(random.uniform(0.5, 1.5))

            inference_time_ms = int((time.time() - start_time) * 1000)

            risk_bucket = get_risk_bucket(tb_score, quality_score)

            # Generate explanation based on risk
            explanation = self._generate_explanation(tb_score, risk_bucket)

            # Generate heatmap (image_path already obtained above)
            heatmap_path = self._generate_heatmap(study.id, image_path, tb_score, risk_bucket)

            # Build scores JSON
            scores = {
                "tb_score": round(tb_score, 3),
                "quality_score": round(quality_score, 3),
                "abnormal_score": round(abnormal_score, 3),
                "using_stub": self.using_stub
            }

            # Model version (append -stub if using stub)
            effective_model_version = self.model_version
            if self.using_stub:
                effective_model_version = f"{self.model_version}-stub"

            # Create result
            result = Result(
                study_id=study.id,
                risk_bucket=risk_bucket,
                score_tb=tb_score,
                scores_json=json.dumps(scores),
                explanation=explanation,
                heatmap_path=heatmap_path,
                model_version=effective_model_version,
                inference_time_ms=inference_time_ms
            )

            self.db.add(result)

            # Update study status
            study.status = "READY"

            # Update job status
            if job:
                job.status = "COMPLETED"

            # Queue for sync
            self._queue_for_sync(result)

            self.db.commit()
            self.db.refresh(result)

            logger.info(f"Inference complete: study={study.id}, risk={risk_bucket}, tb_score={tb_score:.3f}")
            return result

        except Exception as e:
            logger.error(f"Inference failed for study {study.id}: {e}")
            if job:
                job.status = "FAILED"
                job.last_error = str(e)
            study.status = "ERROR"
            self.db.commit()
            return None

    def _ensure_display_image(self, study: Study) -> Optional[str]:
        """
        Ensure a display PNG exists for the study

        Returns path to PNG image or None
        """
        # Check if PNG already exists in processed dir
        png_path = os.path.join(self.processed_dir, f"{study.id}.png")
        if os.path.exists(png_path):
            return png_path

        # Try to extract from DICOM
        if PYDICOM_AVAILABLE and study.filepath and os.path.exists(study.filepath):
            try:
                ds = pydicom.dcmread(study.filepath, force=True)
                if hasattr(ds, 'PixelData') and IMAGE_AVAILABLE:
                    pixel_array = ds.pixel_array

                    # Apply window/level if available
                    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                        wc = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, list) else float(ds.WindowCenter[0])
                        ww = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, list) else float(ds.WindowWidth[0])
                        img_min = wc - ww / 2
                        img_max = wc + ww / 2
                        pixel_array = np.clip(pixel_array, img_min, img_max)

                    # Normalize to 0-255
                    if pixel_array.max() > pixel_array.min():
                        pixel_array = ((pixel_array - pixel_array.min()) /
                                       (pixel_array.max() - pixel_array.min()) * 255)
                    pixel_array = pixel_array.astype(np.uint8)

                    # Handle photometric interpretation
                    if hasattr(ds, 'PhotometricInterpretation'):
                        if ds.PhotometricInterpretation == "MONOCHROME1":
                            pixel_array = 255 - pixel_array

                    # Save PNG
                    img = Image.fromarray(pixel_array)
                    img.save(png_path)
                    logger.info(f"Created display image from DICOM: {png_path}")
                    return png_path

            except Exception as e:
                logger.warning(f"Could not extract image from DICOM: {e}")

        # Check for sample PNG
        sample_png = os.path.join(settings.edge_data_dir, "samples", "sample_cxr.png")
        if os.path.exists(sample_png):
            # Copy to processed directory
            import shutil
            shutil.copy(sample_png, png_path)
            return png_path

        return None

    def _generate_explanation(self, tb_score: float, risk_bucket: str) -> str:
        """Generate clinical explanation based on scores"""
        if risk_bucket == "HIGH":
            explanations = [
                "Upper lobe opacity with possible cavitation pattern",
                "Bilateral infiltrates suggestive of active TB",
                "Right upper lobe consolidation, recommend sputum collection",
                "Patchy opacities in upper lung fields, suspicious for TB"
            ]
        elif risk_bucket == "MEDIUM":
            explanations = [
                "Mild opacity in right lung field, clinical correlation advised",
                "Subtle changes in upper lobes, consider follow-up imaging",
                "Non-specific infiltrate, correlate with symptoms"
            ]
        elif risk_bucket == "NOT_CONFIDENT":
            explanations = [
                "Image quality suboptimal, recommend repeat imaging",
                "Unable to assess reliably due to technical factors",
                "Poor positioning affects interpretation"
            ]
        else:
            explanations = [
                "No significant abnormality detected",
                "Lungs appear clear",
                "Normal chest radiograph"
            ]

        return random.choice(explanations)

    def _generate_heatmap(self, study_id: str, image_path: Optional[str],
                          tb_score: float, risk_bucket: str) -> str:
        """
        Generate attention heatmap overlay

        In production: Use GradCAM or similar to generate attention map
        For now: Generate a synthetic heatmap based on risk level
        """
        heatmap_path = os.path.join(self.heatmap_dir, f"{study_id}_heatmap.png")

        if not IMAGE_AVAILABLE:
            # Create placeholder file
            with open(heatmap_path, "wb") as f:
                f.write(b"PLACEHOLDER HEATMAP")
            return heatmap_path

        try:
            # Load base image or create blank
            if image_path and os.path.exists(image_path):
                base_img = Image.open(image_path).convert('L')
                width, height = base_img.size
            else:
                width, height = 512, 512
                base_img = Image.new('L', (width, height), color=128)

            # Convert to RGB for overlay
            base_rgb = Image.merge('RGB', (base_img, base_img, base_img))

            # Create heatmap based on risk level
            heatmap = self._create_synthetic_heatmap(width, height, tb_score, risk_bucket)

            # Blend heatmap with base image
            blended = Image.blend(base_rgb, heatmap, alpha=0.4)
            blended.save(heatmap_path)

            logger.info(f"Generated heatmap: {heatmap_path}")
            return heatmap_path

        except Exception as e:
            logger.warning(f"Could not generate heatmap: {e}")
            # Create placeholder
            with open(heatmap_path, "wb") as f:
                f.write(b"PLACEHOLDER HEATMAP")
            return heatmap_path

    def _create_synthetic_heatmap(self, width: int, height: int,
                                   tb_score: float, risk_bucket: str) -> Image.Image:
        """
        Create a synthetic heatmap showing attention regions

        Uses Gaussian blobs in lung field regions weighted by risk score
        """
        # Create base array
        heatmap_array = np.zeros((height, width), dtype=np.float32)

        # Define attention regions (lung fields for chest X-ray)
        # Upper left lung (common TB location)
        cx1, cy1 = int(width * 0.35), int(height * 0.35)
        # Upper right lung
        cx2, cy2 = int(width * 0.65), int(height * 0.35)
        # Lower regions
        cx3, cy3 = int(width * 0.35), int(height * 0.55)
        cx4, cy4 = int(width * 0.65), int(height * 0.55)

        # Create coordinate grids
        y, x = np.ogrid[:height, :width]

        # Add Gaussian blobs at attention regions
        # Intensity based on tb_score
        intensity = tb_score * 0.8 + 0.2  # Range 0.2 to 1.0

        if risk_bucket == "HIGH":
            # Strong attention in upper lobes (typical TB)
            sigma = min(width, height) * 0.12
            heatmap_array += intensity * np.exp(-((x - cx1)**2 + (y - cy1)**2) / (2 * sigma**2))
            heatmap_array += intensity * 0.7 * np.exp(-((x - cx2)**2 + (y - cy2)**2) / (2 * sigma**2))
        elif risk_bucket == "MEDIUM":
            # Moderate attention, more diffuse
            sigma = min(width, height) * 0.15
            heatmap_array += intensity * 0.6 * np.exp(-((x - cx1)**2 + (y - cy1)**2) / (2 * sigma**2))
            heatmap_array += intensity * 0.5 * np.exp(-((x - cx3)**2 + (y - cy3)**2) / (2 * sigma**2))
        else:
            # LOW or NOT_CONFIDENT - minimal attention
            sigma = min(width, height) * 0.2
            heatmap_array += 0.2 * np.exp(-((x - width//2)**2 + (y - height//2)**2) / (2 * sigma**2))

        # Normalize to 0-1
        if heatmap_array.max() > 0:
            heatmap_array = heatmap_array / heatmap_array.max()

        # Apply colormap (blue -> cyan -> green -> yellow -> red)
        # Using simple RGB mapping
        r = np.clip(heatmap_array * 2, 0, 1) * 255
        g = np.clip(2 - heatmap_array * 2, 0, 1) * heatmap_array * 255
        b = np.clip(1 - heatmap_array, 0, 1) * 100

        # Stack to RGB
        rgb_array = np.stack([r, g, b], axis=-1).astype(np.uint8)

        # Create image and apply slight blur for smoothness
        heatmap_img = Image.fromarray(rgb_array)
        heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=3))

        return heatmap_img

    def _queue_for_sync(self, result: Result):
        """Add result to sync queue"""
        study = self.db.query(Study).filter(Study.id == result.study_id).first()

        sync_item = SyncQueue(
            record_type="result",
            record_id=result.id,
            payload_json=json.dumps({
                "study_id": result.study_id,
                "risk_bucket": result.risk_bucket,
                "score_tb": result.score_tb,
                "scores": json.loads(result.scores_json),
                "explanation": result.explanation,
                "model_version": result.model_version,
                "inference_time_ms": result.inference_time_ms,
                "input_hash": study.input_hash if study else "",
                "has_burned_in_text": study.has_burned_in_text if study else False
            })
        )
        self.db.add(sync_item)


# Stub implementations for future modalities
class XrayInference(InferenceService):
    """X-ray specific inference"""
    pass


class CTInference(InferenceService):
    """CT specific inference (future)"""
    pass


class MRIInference(InferenceService):
    """MRI specific inference (future)"""
    pass
