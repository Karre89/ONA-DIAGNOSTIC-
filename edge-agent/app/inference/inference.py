import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Optional

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


class InferenceService:
    """
    Inference Service

    In production, this would:
    1. Load ONNX/PyTorch model
    2. Preprocess image (resize, normalize, windowing)
    3. Run inference
    4. Generate heatmap (GradCAM)
    5. Return scores and explanation

    For v1, we simulate with deterministic random scores based on study ID.
    """

    def __init__(self, db: Session):
        self.db = db
        self.model_version = settings.default_model_version
        self.heatmap_dir = os.path.join(settings.edge_data_dir, "heatmaps")
        self.processed_dir = os.path.join(settings.edge_data_dir, "processed")
        os.makedirs(self.heatmap_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def process_study(self, study: Study) -> Optional[Result]:
        """
        Run inference on a study

        Returns Result if successful, None otherwise
        """
        logger.info(f"Running inference on study: {study.id}")

        # Update job status
        job = self.db.query(Job).filter(Job.study_id == study.id).first()
        if job:
            job.status = "RUNNING"
            job.attempts += 1
            self.db.commit()

        try:
            # Simulate inference time
            start_time = time.time()
            time.sleep(random.uniform(0.5, 2.0))  # Simulate processing
            inference_time_ms = int((time.time() - start_time) * 1000)

            # Generate mock scores
            # Use study.id hash to get consistent scores for same study
            random.seed(hash(study.id))

            tb_score = random.uniform(0, 1)
            quality_score = random.uniform(0.6, 1.0)
            abnormal_score = random.uniform(0, 1)

            # Reset random seed
            random.seed()

            risk_bucket = get_risk_bucket(tb_score, quality_score)

            # Generate explanation based on risk
            explanation = self._generate_explanation(tb_score, risk_bucket)

            # Load or create display image and generate heatmap
            image_path = self._ensure_display_image(study)
            heatmap_path = self._generate_heatmap(study.id, image_path, tb_score, risk_bucket)

            # Build scores JSON
            scores = {
                "tb_score": round(tb_score, 3),
                "quality_score": round(quality_score, 3),
                "abnormal_score": round(abnormal_score, 3)
            }

            # Create result
            result = Result(
                study_id=study.id,
                risk_bucket=risk_bucket,
                score_tb=tb_score,
                scores_json=json.dumps(scores),
                explanation=explanation,
                heatmap_path=heatmap_path,
                model_version=self.model_version,
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
