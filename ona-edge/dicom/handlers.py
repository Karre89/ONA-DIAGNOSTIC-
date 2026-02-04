"""
DICOM Event Handlers
Process incoming images and trigger AI pipeline
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import hashlib

logger = logging.getLogger('ona.dicom.handlers')


def generate_scan_id():
    """Generate unique scan ID"""
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    return f"SCAN-{timestamp}"


def handle_store(event, inference_callback=None):
    """
    Handle C-STORE request (incoming image)

    Returns:
        0x0000 = Success
        0xC001 = Processing failure
    """
    try:
        ds = event.dataset
        ds.file_meta = event.file_meta

        # Generate unique ID for this scan
        scan_id = generate_scan_id()
        timestamp = datetime.utcnow().isoformat()

        # Extract metadata before de-identification
        metadata = {
            'scan_id': scan_id,
            'timestamp': timestamp,
            'modality': getattr(ds, 'Modality', 'UNKNOWN'),
            'study_date': getattr(ds, 'StudyDate', ''),
            'body_part': getattr(ds, 'BodyPartExamined', 'CHEST'),
            'sop_class': str(ds.file_meta.MediaStorageSOPClassUID),
        }

        # De-identify: Remove patient information
        ds = remove_phi(ds)

        # Save to local storage
        from config import STORAGE_DIR
        scan_dir = Path(STORAGE_DIR) / scan_id
        scan_dir.mkdir(parents=True, exist_ok=True)
        filepath = scan_dir / 'image.dcm'
        ds.save_as(str(filepath))

        logger.info(f"Received scan: {scan_id} | Modality: {metadata['modality']}")

        # Queue for AI inference (non-blocking)
        if inference_callback:
            inference_callback(scan_id, str(filepath), metadata)

        return 0x0000  # Success

    except Exception as e:
        logger.error(f"Failed to process DICOM: {e}")
        return 0xC001  # Processing failure


def handle_echo(event):
    """
    Handle C-ECHO request (connectivity test)
    Always return success - this is like a DICOM "ping"
    """
    logger.debug(f"C-ECHO from {event.assoc.requestor.address}")
    return 0x0000


def remove_phi(ds):
    """
    Remove Patient Health Information from DICOM dataset
    HIPAA Safe Harbor compliant
    """
    # Tags to remove
    phi_tags = [
        'PatientName', 'PatientID', 'PatientBirthDate',
        'PatientBirthTime', 'OtherPatientIDs', 'OtherPatientNames',
        'InstitutionName', 'InstitutionAddress', 'StationName',
        'ReferringPhysicianName', 'PerformingPhysicianName',
        'OperatorsName', 'AccessionNumber'
    ]

    for tag in phi_tags:
        if hasattr(ds, tag):
            delattr(ds, tag)

    # Replace with anonymous values
    ds.PatientName = "ANONYMOUS"
    ds.PatientID = "ONA_ANON"

    # Remove private tags
    ds.remove_private_tags()

    return ds
