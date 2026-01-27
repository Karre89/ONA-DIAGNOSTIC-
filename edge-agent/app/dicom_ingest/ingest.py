import os
import hashlib
import logging
import threading
from datetime import datetime
from typing import Optional, Callable

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.models import Study, Job

logger = logging.getLogger(__name__)

# DICOM imports - wrapped in try/except for graceful degradation
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    from pynetdicom import AE, evt, AllStoragePresentationContexts
    from pynetdicom.sop_class import Verification
    DICOM_AVAILABLE = True
    PYNETDICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    PYNETDICOM_AVAILABLE = False
    logger.warning("pydicom/pynetdicom not available - DICOM SCP disabled")


class DicomIngestService:
    """
    DICOM Ingest Service - handles incoming studies via C-STORE SCP

    Features:
    1. Listens on port 104 (or configured port) for DICOM C-STORE
    2. Receives and stores DICOM files
    3. Extracts metadata from DICOM headers
    4. Groups files into studies
    5. Creates stable study IDs based on DICOM StudyInstanceUID
    """

    def __init__(self, db: Session):
        self.db = db
        self.incoming_dir = os.path.join(settings.edge_data_dir, "incoming")
        self.processed_dir = os.path.join(settings.edge_data_dir, "processed")

        # Ensure directories exist
        os.makedirs(self.incoming_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def create_study_from_dicom(self, ds: 'Dataset', filepath: str) -> Study:
        """Create a study from a DICOM dataset"""

        # Extract DICOM metadata
        study_instance_uid = str(getattr(ds, 'StudyInstanceUID', generate_uid()))
        modality = str(getattr(ds, 'Modality', 'OT'))
        patient_id = str(getattr(ds, 'PatientID', 'UNKNOWN'))
        study_date = str(getattr(ds, 'StudyDate', datetime.utcnow().strftime('%Y%m%d')))

        # Generate stable study ID from StudyInstanceUID
        study_hash = hashlib.md5(study_instance_uid.encode()).hexdigest()[:6].upper()
        study_id = f"STU-{study_date}-{study_hash}"

        # Compute input hash for deduplication
        input_hash = self._compute_hash(filepath)

        # Check for existing study with same StudyInstanceUID
        existing = self.db.query(Study).filter(Study.id == study_id).first()
        if existing:
            logger.info(f"Existing study found: {existing.id}")
            # Update filepath if new instance
            if existing.filepath != filepath:
                existing.filepath = filepath
                self.db.commit()
            return existing

        # Create new study
        study = Study(
            id=study_id,
            modality=modality,
            filepath=filepath,
            input_hash=input_hash,
            status="RECEIVED"
        )

        self.db.add(study)
        self.db.commit()
        self.db.refresh(study)

        logger.info(f"Created study from DICOM: {study.id}, modality={modality}")

        # Create processing job
        job = Job(
            study_id=study.id,
            status="PENDING"
        )
        self.db.add(job)
        self.db.commit()

        return study

    def create_study(
        self,
        filepath: str,
        modality: str = "CXR",
        study_id: Optional[str] = None
    ) -> Study:
        """Create a new study record (legacy method for non-DICOM files)"""

        # Try to read as DICOM first
        if DICOM_AVAILABLE and os.path.exists(filepath):
            try:
                ds = pydicom.dcmread(filepath, force=True)
                return self.create_study_from_dicom(ds, filepath)
            except Exception as e:
                logger.debug(f"Not a valid DICOM file: {e}")

        # Fall back to simple study creation
        input_hash = self._compute_hash(filepath) if os.path.exists(filepath) else hashlib.md5(filepath.encode()).hexdigest()

        # Check for duplicate
        existing = self.db.query(Study).filter(Study.input_hash == input_hash).first()
        if existing:
            logger.info(f"Duplicate study detected: {existing.id}")
            return existing

        # Create study
        if study_id is None:
            study_id = f"STU-{datetime.utcnow().strftime('%Y%m%d')}-{hashlib.md5(str(datetime.utcnow().timestamp()).encode()).hexdigest()[:6].upper()}"

        study = Study(
            id=study_id,
            modality=modality,
            filepath=filepath,
            input_hash=input_hash,
            status="RECEIVED"
        )

        self.db.add(study)
        self.db.commit()
        self.db.refresh(study)

        logger.info(f"Created study: {study.id}")

        # Create processing job
        job = Job(
            study_id=study.id,
            status="PENDING"
        )
        self.db.add(job)
        self.db.commit()

        return study

    def _compute_hash(self, filepath: str) -> str:
        """Compute MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return hashlib.md5(filepath.encode()).hexdigest()


class DicomSCP:
    """
    DICOM C-STORE SCP (Service Class Provider)

    Listens for incoming DICOM images from modalities (X-ray machines, CT scanners, etc.)
    and stores them for processing.
    """

    def __init__(self, db_session_factory: Callable, port: int = 104, ae_title: str = "ONA_EDGE"):
        self.db_session_factory = db_session_factory
        self.port = port
        self.ae_title = ae_title
        self.ae = None
        self._server_thread = None
        self._running = False

        if not DICOM_AVAILABLE:
            logger.error("DICOM libraries not available - SCP cannot start")
            return

        # Create Application Entity
        self.ae = AE(ae_title=ae_title)

        # Accept all storage SOP classes
        self.ae.supported_contexts = AllStoragePresentationContexts

        # Also support verification (C-ECHO)
        self.ae.add_supported_context(Verification)

    def handle_store(self, event: 'evt.Event') -> int:
        """Handle a C-STORE request"""
        try:
            ds = event.dataset
            ds.file_meta = event.file_meta

            # Generate filename from SOP Instance UID
            sop_instance_uid = str(ds.SOPInstanceUID)
            study_instance_uid = str(getattr(ds, 'StudyInstanceUID', 'unknown'))

            # Create directory structure: incoming/{StudyInstanceUID}/
            study_dir = os.path.join(
                settings.edge_data_dir,
                "incoming",
                study_instance_uid
            )
            os.makedirs(study_dir, exist_ok=True)

            # Save DICOM file
            filepath = os.path.join(study_dir, f"{sop_instance_uid}.dcm")
            ds.save_as(filepath)

            logger.info(f"Received DICOM: {filepath}")

            # Create study record
            db = self.db_session_factory()
            try:
                service = DicomIngestService(db)
                study = service.create_study_from_dicom(ds, filepath)
                logger.info(f"Study created/updated: {study.id}")
            finally:
                db.close()

            # Return success
            return 0x0000  # Success

        except Exception as e:
            logger.error(f"Error handling C-STORE: {e}")
            return 0xC000  # Error

    def handle_echo(self, event: 'evt.Event') -> int:
        """Handle a C-ECHO request (verification)"""
        logger.info(f"Received C-ECHO from {event.assoc.requestor.ae_title}")
        return 0x0000  # Success

    def start(self):
        """Start the DICOM SCP server"""
        if not DICOM_AVAILABLE or self.ae is None:
            logger.warning("DICOM SCP not started - libraries not available")
            return

        if self._running:
            logger.warning("DICOM SCP already running")
            return

        # Set up event handlers
        handlers = [
            (evt.EVT_C_STORE, self.handle_store),
            (evt.EVT_C_ECHO, self.handle_echo),
        ]

        def run_server():
            logger.info(f"Starting DICOM SCP on port {self.port} (AE Title: {self.ae_title})")
            self.ae.start_server(
                ("0.0.0.0", self.port),
                evt_handlers=handlers,
                block=True
            )

        self._running = True
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        logger.info(f"DICOM SCP started on port {self.port}")

    def stop(self):
        """Stop the DICOM SCP server"""
        if self.ae and self._running:
            self.ae.shutdown()
            self._running = False
            logger.info("DICOM SCP stopped")


def create_sample_study(db: Session) -> Study:
    """Create a sample study with a real DICOM file for testing"""
    service = DicomIngestService(db)

    # Create sample directory
    sample_dir = os.path.join(settings.edge_data_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    sample_path = os.path.join(sample_dir, "sample_cxr.dcm")

    # Create a synthetic DICOM file if pydicom is available
    if DICOM_AVAILABLE:
        try:
            create_sample_dicom(sample_path)
        except Exception as e:
            logger.warning(f"Could not create sample DICOM: {e}")
            # Create placeholder
            if not os.path.exists(sample_path):
                with open(sample_path, "wb") as f:
                    f.write(b"MOCK DICOM DATA - " + str(datetime.utcnow()).encode())
    else:
        # Create placeholder file
        if not os.path.exists(sample_path):
            with open(sample_path, "wb") as f:
                f.write(b"MOCK DICOM DATA - " + str(datetime.utcnow()).encode())

    study = service.create_study(sample_path, modality="CXR")
    return study


def create_sample_dicom(filepath: str):
    """Create a synthetic DICOM chest X-ray file with pixel data"""
    import numpy as np
    from PIL import Image

    # Create synthetic chest X-ray image (512x512 grayscale)
    # Simulate lung fields with darker regions
    img_array = np.ones((512, 512), dtype=np.uint16) * 2000

    # Create lung-like darker regions
    y, x = np.ogrid[:512, :512]

    # Left lung (ellipse)
    left_lung = ((x - 180) ** 2 / 80 ** 2 + (y - 280) ** 2 / 150 ** 2) < 1
    img_array[left_lung] = 800

    # Right lung (ellipse)
    right_lung = ((x - 332) ** 2 / 80 ** 2 + (y - 280) ** 2 / 150 ** 2) < 1
    img_array[right_lung] = 800

    # Heart shadow (center-left, darker)
    heart = ((x - 230) ** 2 / 60 ** 2 + (y - 320) ** 2 / 80 ** 2) < 1
    img_array[heart] = 2500

    # Spine (vertical darker line)
    spine = (abs(x - 256) < 15) & (y > 150) & (y < 480)
    img_array[spine] = 2800

    # Ribs (horizontal curves)
    for i in range(6):
        rib_y = 180 + i * 50
        rib_mask = (abs(y - rib_y) < 5) & (x > 120) & (x < 400)
        img_array[rib_mask] = 2200

    # Add some noise for realism
    noise = np.random.normal(0, 50, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 4095).astype(np.uint16)

    # Create DICOM file metadata
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    # Create the DICOM dataset
    ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient info (will be de-identified later)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "TEST001"
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "M"

    # Study info
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = datetime.utcnow().strftime("%Y%m%d")
    ds.StudyTime = datetime.utcnow().strftime("%H%M%S")
    ds.StudyDescription = "CHEST X-RAY PA"
    ds.AccessionNumber = f"ACC{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    # Series info
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 1
    ds.Modality = "CR"  # Computed Radiography (chest X-ray)

    # Instance info
    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = 1

    # Image info
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.WindowCenter = 2000
    ds.WindowWidth = 4000

    # Add pixel data
    ds.PixelData = img_array.tobytes()

    # Set creation date/time
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Save
    ds.save_as(filepath)
    logger.info(f"Created sample DICOM file: {filepath}")

    # Also save as PNG for display
    png_path = filepath.replace('.dcm', '.png')
    # Normalize to 8-bit for PNG
    img_8bit = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    Image.fromarray(img_8bit).save(png_path)
    logger.info(f"Created sample PNG: {png_path}")

    return ds
