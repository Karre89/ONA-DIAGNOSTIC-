"""
Create a synthetic test DICOM for pipeline testing
No external data needed - generates a chest X-ray-like image
"""

import numpy as np
from pathlib import Path
from datetime import datetime

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, ComputedRadiographyImageStorage
import tempfile


def create_synthetic_cxr(size=512):
    """Generate a synthetic chest X-ray-like image"""

    # Create base image (gray background)
    img = np.ones((size, size), dtype=np.float32) * 180

    # Add noise
    img += np.random.normal(0, 10, (size, size))

    # Create coordinate grids (meshgrid for proper 2D arrays)
    y, x = np.mgrid[:size, :size]
    center_y, center_x = size // 2, size // 2

    # Add lung fields (darker ellipses)
    # Left lung
    left_lung_mask = ((x - center_x + size//6)**2 / (size//5)**2 +
                      (y - center_y)**2 / (size//3)**2) < 1
    img[left_lung_mask] -= 60

    # Right lung
    right_lung_mask = ((x - center_x - size//6)**2 / (size//5)**2 +
                       (y - center_y)**2 / (size//3)**2) < 1
    img[right_lung_mask] -= 60

    # Add heart shadow (brighter)
    heart_mask = ((x - center_x + size//10)**2 / (size//8)**2 +
                  (y - center_y + size//8)**2 / (size//6)**2) < 1
    img[heart_mask] += 30

    # Add spine (bright vertical line)
    spine_mask = np.abs(x - center_x) < size//40
    img[spine_mask] += 40

    # Add ribs (horizontal lines)
    for i in range(-4, 5):
        rib_y = center_y + i * (size // 10)
        rib_mask = (np.abs(y - rib_y) < 3) & (np.abs(x - center_x) < size//3)
        img[rib_mask] += 20

    # Clip to valid range
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def create_test_dicom(output_path: str, add_abnormality: bool = False):
    """Create a valid test DICOM file"""

    # Generate synthetic image
    pixels = create_synthetic_cxr(512)

    if add_abnormality:
        # Add a "lesion" in upper lobe (simulates TB)
        y, x = np.mgrid[:512, :512]
        lesion_y, lesion_x = 150, 200  # Upper left
        lesion_mask = ((x - lesion_x)**2 + (y - lesion_y)**2) < 25**2
        pixels[lesion_mask] = np.clip(pixels[lesion_mask] + 50, 0, 255).astype(np.uint8)
        print("  Added simulated abnormality (upper left)")

    # Create file meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ComputedRadiographyImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create the FileDataset instance
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Add required DICOM attributes
    ds.PatientName = "TEST^SYNTHETIC"
    ds.PatientID = "SYNTH001"
    ds.PatientBirthDate = ""
    ds.PatientSex = ""

    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S")
    ds.StudyDescription = "Synthetic Test"
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = ""
    ds.StudyID = "1"

    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 1
    ds.Modality = "CR"
    ds.SeriesDescription = "Chest X-Ray"

    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.InstanceNumber = 1

    # Image attributes
    ds.Rows = pixels.shape[0]
    ds.Columns = pixels.shape[1]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BodyPartExamined = "CHEST"

    # Set pixel data
    ds.PixelData = pixels.tobytes()

    # Set encoding
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Save
    ds.save_as(output_path, write_like_original=False)

    print(f"✓ Created test DICOM: {output_path}")
    print(f"  Size: {pixels.shape[1]}x{pixels.shape[0]}")
    print(f"  Type: Synthetic chest X-ray")


def main():
    output_dir = Path(__file__).parent.parent / "data" / "test_dicoms"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating test DICOM files...\n")

    # Create normal image
    create_test_dicom(str(output_dir / "test_normal.dcm"), add_abnormality=False)

    # Create abnormal image
    create_test_dicom(str(output_dir / "test_abnormal.dcm"), add_abnormality=True)

    print(f"\n✓ Test files created in: {output_dir}")
    print("\nTo test the pipeline:")
    print("  1. Start edge software:  python main.py")
    print("  2. Send test DICOM:      python -m pynetdicom storescu localhost 11112 data/test_dicoms/test_normal.dcm")
    print("  3. View results:         http://localhost:8080")


if __name__ == "__main__":
    main()
