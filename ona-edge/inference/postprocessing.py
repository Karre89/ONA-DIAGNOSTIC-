"""
Post-processing for AI Results
Generates heatmaps and reports
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Optional


def generate_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int = 0,
    layer_name: str = 'features'
) -> np.ndarray:
    """
    Generate GradCAM heatmap for model explainability

    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        target_class: Class to explain
        layer_name: Name of layer to use for GradCAM

    Returns:
        Heatmap as numpy array (0-255, uint8)
    """
    model.eval()

    # Get the feature layer
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    # Try to find a suitable layer
    target_layer = None
    for name, module in model.named_modules():
        if 'features' in name.lower() or 'layer4' in name.lower():
            target_layer = module
            break

    if target_layer is None:
        # Return blank heatmap if layer not found
        return np.zeros((224, 224), dtype=np.uint8)

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        input_tensor = image_tensor.unsqueeze(0).requires_grad_(True)
        output = model(input_tensor)

        # Backward pass
        model.zero_grad()
        target = output[0, target_class] if output.dim() > 1 else output[0]
        target.backward()

        # Generate heatmap
        if gradients is not None and features is not None:
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * features).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()

            # Normalize
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            cam = (cam * 255).astype(np.uint8)

            # Resize to standard size
            cam = cv2.resize(cam, (224, 224))

            return cam
        else:
            return np.zeros((224, 224), dtype=np.uint8)

    except Exception as e:
        return np.zeros((224, 224), dtype=np.uint8)

    finally:
        fh.remove()
        bh.remove()


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay heatmap on original image

    Args:
        original_image: Original grayscale or RGB image
        heatmap: Heatmap array
        alpha: Blending factor

    Returns:
        Blended image with heatmap overlay
    """
    # Ensure original is 3-channel
    if len(original_image.shape) == 2:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_image

    # Resize heatmap to match original
    heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))

    # Apply colormap (red = high probability)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Blend
    blended = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    return blended


def save_result_images(
    scan_id: str,
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    output_dir: str
) -> dict:
    """
    Save result images to disk

    Returns:
        Dictionary with file paths
    """
    from pathlib import Path

    output_path = Path(output_dir) / scan_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Save images
    original_path = output_path / 'original.png'
    heatmap_path = output_path / 'heatmap.png'
    overlay_path = output_path / 'overlay.png'

    cv2.imwrite(str(original_path), original)
    cv2.imwrite(str(heatmap_path), heatmap)
    cv2.imwrite(str(overlay_path), overlay)

    return {
        'original': str(original_path),
        'heatmap': str(heatmap_path),
        'overlay': str(overlay_path)
    }
