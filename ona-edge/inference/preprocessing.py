"""
Image Preprocessing for Medical AI
Handles DICOM loading and normalization
"""

import numpy as np
import torch
from pydicom import dcmread
from PIL import Image
import cv2
from typing import Tuple


def preprocess_dicom(
    dicom_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess DICOM for inference

    Args:
        dicom_path: Path to DICOM file
        target_size: Output image size (H, W)

    Returns:
        Tuple of (normalized tensor, original image array)
    """
    # Load DICOM
    ds = dcmread(dicom_path)

    # Extract pixel data
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply modality-specific processing
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

    # Handle photometric interpretation
    if hasattr(ds, 'PhotometricInterpretation'):
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            # Invert: white = air, black = bone
            pixel_array = pixel_array.max() - pixel_array

    # Normalize to 0-255 range
    pixel_array = normalize_to_uint8(pixel_array)

    # Keep original for display
    original = pixel_array.copy()

    # Resize for model
    resized = cv2.resize(pixel_array, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert to 3-channel (models expect RGB)
    if len(resized.shape) == 2:
        resized = np.stack([resized, resized, resized], axis=0)
    else:
        resized = np.transpose(resized, (2, 0, 1))

    # Normalize for neural network (ImageNet stats)
    tensor = torch.from_numpy(resized).float()
    tensor = tensor / 255.0

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor, original


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize array to 0-255 uint8 range"""
    array = array - array.min()
    if array.max() > 0:
        array = array / array.max()
    array = (array * 255).astype(np.uint8)
    return array


def load_image_file(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess a regular image file (PNG, JPG)
    For testing without DICOM
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    original = np.array(img)

    # Resize
    img_resized = img.resize(target_size, Image.BILINEAR)
    resized = np.array(img_resized)

    # Convert to 3-channel
    resized_3ch = np.stack([resized, resized, resized], axis=0)

    # Normalize
    tensor = torch.from_numpy(resized_3ch).float() / 255.0

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor, original
