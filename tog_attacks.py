"""Projected-gradient TOG attacks for compatible object-detector attack models."""

from __future__ import annotations

from typing import Literal
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np

from tog_model import YOLOv3TOGModel
from utils import generate_attack_targets

from PIL import Image
from utils import letterbox_image


def _validate_attack_inputs(image: np.ndarray, num_iterations: int, epsilon: float, step_size: float) -> None:
    """Validate common attack parameters before running iterative updates."""
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("image must have shape (batch, height, width, 3).")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if epsilon < 0 or step_size <= 0:
        raise ValueError("epsilon must be non-negative and step_size must be positive.")


def _initialize_adversarial_image(image: np.ndarray, epsilon: float) -> np.ndarray:
    """Randomly initialize an adversarial image inside the L-infinity constraint."""
    perturbation = np.random.uniform(-epsilon, epsilon, size=image.shape)
    return np.clip(image + perturbation, 0.0, 1.0)


def _project(image: np.ndarray, adversarial_image: np.ndarray, epsilon: float) -> np.ndarray:
    """Project an adversarial image into the valid pixel and L-infinity ranges."""
    perturbation = np.clip(adversarial_image - image, -epsilon, epsilon)
    return np.clip(image + perturbation, 0.0, 1.0)


def tog_vanishing(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Generate a TOG object-vanishing adversarial image."""
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    adversarial_image = _initialize_adversarial_image(image, epsilon)
    for _ in range(num_iterations):
        grad = victim.compute_object_vanishing_gradient(adversarial_image)
        adversarial_image = _project(
            image, 
            adversarial_image - step_size * np.sign(grad), 
            epsilon
        )
    return adversarial_image


def tog_fabrication(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Generate a TOG object-fabrication adversarial image."""
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    adversarial_image = _initialize_adversarial_image(image, epsilon)
    for _ in range(num_iterations):
        grad = victim.compute_object_fabrication_gradient(adversarial_image)
        adversarial_image = _project(
            image, 
            adversarial_image - step_size * np.sign(grad), 
            epsilon
        )
    return adversarial_image


def tog_mislabeling(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    target: Literal["most_likely", "least_likely"] = "least_likely",
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Generate a targeted TOG object-mislabeling adversarial image."""
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    
    detections = victim.detect(image)
    if len(detections) == 0:
        raise ValueError("No objects detected to perform mislabeling attack.")

    # Đã sửa: Truyền đúng tham số mode và confidence_threshold
    target_detections = generate_attack_targets(
        detections, 
        mode=target, 
        confidence_threshold=victim.confidence_threshold
    )
    
    adversarial_image = _initialize_adversarial_image(image, epsilon)
    
    for _ in range(num_iterations):
        # Đã sửa: Đổi tên hàm thành compute_object_mislabeling_gradient
        grad = victim.compute_object_mislabeling_gradient(adversarial_image, target_detections)
        adversarial_image = _project(
            image, 
            adversarial_image - step_size * np.sign(grad), 
            epsilon
        )
    return adversarial_image


def tog_untargeted(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Generate an untargeted TOG adversarial image."""
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    
    detections = victim.detect(image)
    if len(detections) == 0:
        raise ValueError("No benign objects detected in the image.")
        
    adversarial_image = _initialize_adversarial_image(image, epsilon)
    
    for _ in range(num_iterations):
        # Đã sửa: Đổi tên hàm thành compute_object_untargeted_gradient
        grad = victim.compute_object_untargeted_gradient(adversarial_image, detections)
        adversarial_image = _project(
            image, 
            adversarial_image - step_size * np.sign(grad), 
            epsilon
        )
    return adversarial_image


def tog_universal(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
    data_path: str = '',
    n_train_samples: int = 100
) -> np.ndarray:
    """Generate a universal adversarial perturbation (UAP) noise array."""
    fpaths_train = [os.path.join(data_path, file.name) for file in Path(data_path).iterdir() if file.is_file()]
    
    if not fpaths_train:
        raise ValueError(f"No images found in path: {data_path}")

    random.shuffle(fpaths_train)
    fpaths_train = fpaths_train[:n_train_samples]

    noise = np.zeros_like(image)

    for epoch in range(num_iterations):
        pbar = tqdm(fpaths_train)
        pbar.set_description(f'Epoch {epoch + 1}/{num_iterations}')

        for fpath in pbar:
            # Tải ảnh và chuẩn hóa
            img, _ = letterbox_image(Image.open(fpath), size=victim.model_image_size)
            adv_img = np.clip(img + noise, 0.0, 1.0)
            grad = victim.compute_object_vanishing_gradient(adv_img)
            noise = np.clip(noise - step_size * np.sign(grad), -epsilon, epsilon)
        
    return noise