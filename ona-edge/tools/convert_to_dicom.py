"""
Convert PNG/JPG X-ray images to DICOM format for testing
Usage: python convert_to_dicom.py input.png output.dcm
"""

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import numpy as np
from PIL import Image
import sys
from pathlib import Path
from datetime import datetime


def png_to_dicom(png_path: str, output_path: str, patient_id: str = "TEST001"):
    """Convert PNG/JPG image to DICOM format"""

    # Load image as grayscale
    img = Image.open(png_path).convert('L')
    pixels = np.array(img)

    # Create file meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1.1'  # CR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    # Create dataset
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Patient info (de-identified for testing)
    ds.PatientName = "TEST^PATIENT"
    ds.PatientID = patient_id
    ds.PatientBirthDate = ""
    ds.PatientSex = ""

    # Study info
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S")
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = ""
    ds.StudyDescription = "TB Screening Test"

    # Series info
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 1
    ds.Modality = "CR"  # Computed Radiography
    ds.SeriesDescription = "Chest X-Ray"

    # Instance info
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.InstanceNumber = 1

    # Image info
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0  # Unsigned
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BodyPartExamined = "CHEST"

    # Pixel data
    ds.PixelData = pixels.tobytes()

    # Save
    ds.save_as(output_path)
    print(f"✓ Created: {output_path}")
    print(f"  Size: {pixels.shape[1]}x{pixels.shape[0]}")
    print(f"  Patient ID: {patient_id}")


def batch_convert(input_dir: str, output_dir: str, limit: int = 10):
    """Convert multiple images"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    images = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))

    print(f"Found {len(images)} images")
    print(f"Converting first {limit}...\n")

    for i, img_path in enumerate(images[:limit]):
        patient_id = f"TEST{i+1:03d}"
        output_file = output_path / f"test_{i+1:03d}.dcm"
        png_to_dicom(str(img_path), str(output_file), patient_id)

    print(f"\n✓ Converted {min(limit, len(images))} images to {output_dir}")


def send_dicom(dicom_path: str, host: str = '127.0.0.1', port: int = 11112):
    """Send DICOM to local ONA Edge listener"""
    from pynetdicom import AE
    from pydicom import dcmread

    ds = dcmread(dicom_path, force=True)
    ae = AE(ae_title='TEST_SCU')
    ae.add_requested_context(ds.SOPClassUID)

    assoc = ae.associate(host, port, ae_title='ONA_EDGE')
    if assoc.is_established:
        status = assoc.send_c_store(ds)
        assoc.release()
        if status.Status == 0:
            print(f"[OK] Sent to ONA Edge listener")
            return True
        else:
            print(f"[FAIL] C-STORE failed: {status.Status}")
    else:
        print("[FAIL] Could not connect to DICOM listener")
    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Convert:       python convert_to_dicom.py input.png output.dcm")
        print("  Convert+Send:  python convert_to_dicom.py input.png output.dcm --send")
        print("  Batch:         python convert_to_dicom.py --batch input_dir/ output_dir/ [limit]")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        batch_convert(sys.argv[2], sys.argv[3], limit)
    else:
        send_after = "--send" in sys.argv
        args = [a for a in sys.argv[1:] if a != "--send"]
        if len(args) < 2:
            args.append(args[0].rsplit('.', 1)[0] + '.dcm')

        png_to_dicom(args[0], args[1])

        if send_after:
            send_dicom(args[1])
