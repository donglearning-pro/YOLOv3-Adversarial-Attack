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
    return np.clip(image+perturbation,0.0,1.0)        # TODO: clip the image + perturbation by 0.0 and 1.0. Use np.clip


def _project(image: np.ndarray, adversarial_image: np.ndarray, epsilon: float) -> np.ndarray:
    """Project an adversarial image into the valid pixel and L-infinity ranges."""
    perturbation = np.clip(adversarial_image - image, -epsilon, epsilon)
    return np.clip(image+perturbation,0.0,1.0)          # TODO: clip the image + perturbation by 0.0 and 1.0. Use np.clip


def tog_vanishing(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Generate a TOG object-vanishing adversarial image.

    Repeat the following for ``num_iterations``:
       - Compute the object-vanishing gradient.
       - Use the sign of the gradient to update the image in the direction
         that minimizes the vanishing objective.
       - Project the result back into the epsilon-constrained region and
         the valid pixel range ``[0, 1]``.
    """
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    adversarial_image = _initialize_adversarial_image(image, epsilon)

    for _ in range(num_iterations):
        # 1. Compute the object-vanishing gradient
        # (Assuming the victim model exposes this method, which is standard for TOG models)
        gradient = victim.compute_object_vanishing_gradient(adversarial_image)
        
        # 2. Use the sign of the gradient to update the image (minimize objective = subtract gradient)
        adversarial_image = adversarial_image - step_size * np.sign(gradient)
        
        # 3. Project the result back into the epsilon-constrained region (L-infinity norm)
        adversarial_image = np.clip(adversarial_image, image - epsilon, image + epsilon)
        
        # 4. Project into the valid pixel range [0, 1]
        adversarial_image = np.clip(adversarial_image, 0.0, 1.0)
        # TODO: update the adversarial image by calculating the object vanishing gradiant
        
    return adversarial_image


def tog_fabrication(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Generate a TOG object-fabrication adversarial image.

    Repeat the following for ``num_iterations``:
       - Compute the object-fabrication gradient.
       - Update the image using the sign of the gradient in the direction
         that minimizes the fabrication objective.
       - Project the image back into the valid L-infinity and pixel ranges.

    The structure is similar to the vanishing attack, but it must call the
    fabrication-gradient method provided by ``victim``.
    """
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    # create random noise for image
    adversarial_image = _initialize_adversarial_image(image,epsilon)# TODO: initialize the random noisy image
    for _ in range(num_iterations):
        # TODO: student fill here
        gradient = victim.compute_object_fabrication_gradient(adversarial_image)
        # gradient = victim.compute_object_fabrication_gradient(image)
        adversarial_image = adversarial_image - step_size * np.sign(gradient)
        adversarial_image = np.clip(adversarial_image, image - epsilon, image + epsilon)
        adversarial_image = np.clip(adversarial_image, 0.0, 1.0)
    return adversarial_image

def tog_mislabeling(
    victim: 'YOLOv3TOGModel',
    image: np.ndarray,
    target: Literal["most_likely", "least_likely"],
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Generate a targeted TOG object-mislabeling adversarial image."""
    
    # Validate the common attack parameters
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    
    # 1. Run the detector once on the original image
    detections = victim.detect(image)
    if detections is None or len(detections) == 0:
        raise ValueError("No objects detected in the initial image to mislabel.")
        
    # 2 & 3. Convert initial detections into targeted detections using the victim's confidence threshold
    target_detections = generate_attack_targets(
        detections=detections, 
        mode=target, 
        confidence_threshold=victim.confidence_threshold # <--- THIS FIXES THE ERROR
    )
    
    # 4. Randomly initialize the adversarial image inside the epsilon bound
    adversarial_image = _initialize_adversarial_image(image, epsilon)
    
    # 5. Iterative optimization
    for _ in range(num_iterations):
        # Compute the mislabeling gradient using the fixed target detections
        gradient = victim.compute_object_mislabeling_gradient(adversarial_image, target_detections) 
        
        # Apply a signed-gradient update (minimizing the mislabeling loss = subtracting gradient)
        adversarial_image = adversarial_image - step_size * np.sign(gradient)
        
        # Project the result into the valid epsilon-constrained perturbation range
        adversarial_image = np.clip(adversarial_image, image - epsilon, image + epsilon)
        
        # Project the result into the valid pixel range [0, 1]
        adversarial_image = np.clip(adversarial_image, 0.0, 1.0)
        
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