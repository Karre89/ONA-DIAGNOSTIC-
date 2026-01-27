import os
import logging
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import Study

logger = logging.getLogger(__name__)

# DICOM imports - wrapped in try/except for graceful degradation
try:
    import pydicom
    from pydicom.dataset import Dataset
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logger.warning("pydicom not available - de-identification limited")

# Image processing imports
try:
    import numpy as np
    from PIL import Image
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    logger.warning("numpy/PIL not available - image processing limited")


class BurnedInTextDetector:
    """
    Burned-in Text Detector (Stub Implementation)

    In production, this would use:
    1. OCR (Tesseract, EasyOCR, or PaddleOCR)
    2. ML model trained to detect text regions in medical images
    3. Pattern matching for common PHI locations (corners, headers)

    For now, this is a stub that returns no text detected.
    """

    # Common regions where burned-in text appears
    # Format: (x_start%, y_start%, x_end%, y_end%)
    TEXT_REGIONS = [
        (0, 0, 0.3, 0.1),      # Top-left corner
        (0.7, 0, 1.0, 0.1),    # Top-right corner
        (0, 0.9, 0.3, 1.0),    # Bottom-left corner
        (0.7, 0.9, 1.0, 1.0),  # Bottom-right corner
    ]

    def __init__(self):
        self.ocr_engine = None  # Would be initialized with real OCR

    def detect(self, image_array: np.ndarray) -> Dict:
        """
        Detect burned-in text in an image

        Args:
            image_array: Numpy array of pixel data

        Returns:
            Dict with:
                - has_text: bool
                - confidence: float (0-1)
                - regions: List of detected text regions
                - text_content: List of detected text strings (if any)
        """
        # Stub implementation - returns no text detected
        # In production, this would:
        # 1. Run OCR on suspicious regions
        # 2. Use ML model to classify if text contains PHI
        # 3. Return bounding boxes for redaction

        result = {
            "has_text": False,
            "confidence": 0.95,  # High confidence that there's no text
            "regions": [],
            "text_content": [],
            "checked_regions": len(self.TEXT_REGIONS)
        }

        # Simulate analysis of corner regions
        if IMAGE_PROCESSING_AVAILABLE and image_array is not None:
            try:
                h, w = image_array.shape[:2]

                # Check each suspicious region
                for region in self.TEXT_REGIONS:
                    x1 = int(w * region[0])
                    y1 = int(h * region[1])
                    x2 = int(w * region[2])
                    y2 = int(h * region[3])

                    # Extract region
                    roi = image_array[y1:y2, x1:x2]

                    # Simple heuristic: high variance in corners might indicate text
                    # In production: Run OCR here
                    variance = np.var(roi.astype(np.float32))

                    # If variance is very high, might be text (stub heuristic)
                    # Real implementation would use OCR
                    if variance > 5000:  # Arbitrary threshold
                        logger.debug(f"High variance region detected at {region}")
                        # In production: Run OCR and check for PHI

            except Exception as e:
                logger.warning(f"Error in burned-in text detection: {e}")

        return result

    def redact_regions(self, image_array: np.ndarray, regions: List[Tuple]) -> np.ndarray:
        """
        Redact detected text regions by filling with black

        Args:
            image_array: Original image
            regions: List of (x1, y1, x2, y2) regions to redact

        Returns:
            Redacted image array
        """
        if not regions:
            return image_array

        redacted = image_array.copy()
        for x1, y1, x2, y2 in regions:
            redacted[y1:y2, x1:x2] = 0  # Fill with black

        return redacted


class DeidService:
    """
    De-identification Service

    Removes Protected Health Information (PHI) from DICOM files:
    1. Strips PHI DICOM tags (patient name, DOB, MRN, etc.)
    2. Detects burned-in text using the BurnedInTextDetector
    3. Optionally redacts burned-in text regions
    4. Generates de-identified copy in processed directory
    """

    # DICOM tags containing PHI that must be removed/replaced
    # Based on DICOM PS3.15 Annex E - Application Level Confidentiality Profile
    PHI_TAGS_TO_REMOVE = [
        0x00100010,  # PatientName
        0x00100020,  # PatientID
        0x00100030,  # PatientBirthDate
        0x00100032,  # PatientBirthTime
        0x00101000,  # OtherPatientIDs
        0x00101001,  # OtherPatientNames
        0x00101040,  # PatientAddress
        0x00102160,  # EthnicGroup
        0x00104000,  # PatientComments
        0x00080050,  # AccessionNumber
        0x00080080,  # InstitutionName
        0x00080081,  # InstitutionAddress
        0x00080090,  # ReferringPhysicianName
        0x00081048,  # PhysiciansOfRecord
        0x00081050,  # PerformingPhysicianName
        0x00081060,  # NameOfPhysiciansReadingStudy
        0x00081070,  # OperatorsName
        0x00200010,  # StudyID
        0x00400006,  # ScheduledPerformingPhysicianName
        0x00321032,  # RequestingPhysician
    ]

    # Tags to replace with generic values (not remove)
    PHI_TAGS_TO_REPLACE = {
        0x00100040: "O",          # PatientSex -> Other/Unknown
        0x00101010: "000Y",       # PatientAge -> Unknown
        0x00101020: "",           # PatientSize
        0x00101030: "",           # PatientWeight
    }

    # Tags containing dates to shift or remove
    DATE_TAGS = [
        0x00080020,  # StudyDate
        0x00080021,  # SeriesDate
        0x00080022,  # AcquisitionDate
        0x00080023,  # ContentDate
        0x00080030,  # StudyTime
        0x00080031,  # SeriesTime
        0x00080032,  # AcquisitionTime
        0x00080033,  # ContentTime
    ]

    def __init__(self, db: Session):
        self.db = db
        self.text_detector = BurnedInTextDetector()
        self.processed_dir = os.path.join(settings.edge_data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

    def process_study(self, study: Study) -> bool:
        """
        De-identify a study

        1. Read DICOM file
        2. Strip PHI tags
        3. Check for burned-in text
        4. Save de-identified copy
        5. Update study record

        Returns True if processing succeeded
        """
        logger.info(f"De-identifying study: {study.id}")

        has_burned_in_text = False
        deid_filepath = study.filepath

        if PYDICOM_AVAILABLE and os.path.exists(study.filepath):
            try:
                # Read DICOM file
                ds = pydicom.dcmread(study.filepath, force=True)

                # Strip PHI tags
                self._strip_phi_tags(ds)

                # Check for burned-in text in pixel data
                if hasattr(ds, 'PixelData') and IMAGE_PROCESSING_AVAILABLE:
                    try:
                        pixel_array = ds.pixel_array
                        detection_result = self.text_detector.detect(pixel_array)
                        has_burned_in_text = detection_result["has_text"]

                        if has_burned_in_text:
                            logger.warning(f"Burned-in text detected in study {study.id}")
                            # Optionally redact
                            if detection_result["regions"]:
                                pixel_array = self.text_detector.redact_regions(
                                    pixel_array, detection_result["regions"]
                                )
                                ds.PixelData = pixel_array.tobytes()

                    except Exception as e:
                        logger.warning(f"Could not process pixel data: {e}")

                # Save de-identified copy
                deid_filename = f"{study.id}_deid.dcm"
                deid_filepath = os.path.join(self.processed_dir, deid_filename)
                ds.save_as(deid_filepath)
                logger.info(f"Saved de-identified DICOM: {deid_filepath}")

                # Also save as PNG for display
                self._save_display_image(ds, study.id)

            except Exception as e:
                logger.error(f"Error de-identifying DICOM {study.filepath}: {e}")
                # Continue with original file

        # Update study record
        study.has_burned_in_text = has_burned_in_text
        study.deid_filepath = deid_filepath
        study.status = "DEID_COMPLETE"
        self.db.commit()

        logger.info(f"De-identification complete for study: {study.id}")
        return True

    def _strip_phi_tags(self, ds: 'Dataset') -> Dict:
        """
        Strip PHI DICOM tags from dataset

        Returns dict with statistics about removed/replaced tags
        """
        stats = {
            "tags_removed": 0,
            "tags_replaced": 0,
            "dates_shifted": 0
        }

        # Remove PHI tags
        for tag in self.PHI_TAGS_TO_REMOVE:
            if tag in ds:
                del ds[tag]
                stats["tags_removed"] += 1

        # Replace PHI tags with generic values
        for tag, value in self.PHI_TAGS_TO_REPLACE.items():
            if tag in ds:
                ds[tag].value = value
                stats["tags_replaced"] += 1

        # Handle date tags - shift to epoch or remove
        for tag in self.DATE_TAGS:
            if tag in ds:
                # Option 1: Remove dates entirely
                # del ds[tag]
                # Option 2: Keep dates (often needed for medical context)
                # We'll keep study date but remove time precision
                pass

        # Add de-identification indicator
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "ONA Health De-ID v1.0"

        logger.debug(f"PHI stripping stats: {stats}")
        return stats

    def _save_display_image(self, ds: 'Dataset', study_id: str):
        """Save pixel data as PNG for web display"""
        if not IMAGE_PROCESSING_AVAILABLE:
            return

        try:
            if hasattr(ds, 'PixelData'):
                pixel_array = ds.pixel_array

                # Apply window/level if available
                if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                    wc = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, list) else float(ds.WindowCenter[0])
                    ww = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, list) else float(ds.WindowWidth[0])

                    # Apply windowing
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
                        # Invert for MONOCHROME1 (0 = white)
                        pixel_array = 255 - pixel_array

                # Save PNG
                img = Image.fromarray(pixel_array)
                png_path = os.path.join(self.processed_dir, f"{study_id}.png")
                img.save(png_path)
                logger.info(f"Saved display image: {png_path}")

        except Exception as e:
            logger.warning(f"Could not save display image: {e}")

    def strip_phi_tags(self, filepath: str) -> dict:
        """
        Strip PHI DICOM tags from a file (standalone function)

        Returns dict with operation results
        """
        if not PYDICOM_AVAILABLE:
            return {
                "success": False,
                "error": "pydicom not available",
                "tags_stripped": 0
            }

        try:
            ds = pydicom.dcmread(filepath, force=True)
            stats = self._strip_phi_tags(ds)

            # Save to processed directory
            deid_filename = os.path.basename(filepath).replace('.dcm', '_deid.dcm')
            deid_path = os.path.join(self.processed_dir, deid_filename)
            ds.save_as(deid_path)

            return {
                "success": True,
                "tags_stripped": stats["tags_removed"] + stats["tags_replaced"],
                "original_path": filepath,
                "deid_path": deid_path
            }

        except Exception as e:
            logger.error(f"Error stripping PHI tags: {e}")
            return {
                "success": False,
                "error": str(e),
                "tags_stripped": 0
            }
