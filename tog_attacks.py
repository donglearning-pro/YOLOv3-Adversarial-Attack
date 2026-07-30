"""Projected-gradient TOG attacks for compatible object-detector attack models (GPU Accelerated)."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tog_model import YOLOv3TOGModel
from utils import generate_attack_targets, letterbox_image


def _validate_attack_inputs(
    image: Union[np.ndarray, torch.Tensor], num_iterations: int, epsilon: float, step_size: float
) -> None:
    """Kiểm tra tính hợp lệ của các tham số đầu vào."""
    if image.ndim not in (3, 4):
        raise ValueError("image must have 3 or 4 dimensions.")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if epsilon < 0 or step_size <= 0:
        raise ValueError("epsilon must be non-negative and step_size must be positive.")


def _prepare_tensor(
    image: Union[np.ndarray, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Chuyển đổi NumPy NHWC sang PyTorch NCHW Tensor trên GPU."""
    orig_shape = image.shape
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2)
    elif isinstance(image, torch.Tensor):
        tensor = image.to(device=device, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        elif tensor.ndim == 4 and tensor.shape[1] != 3 and tensor.shape[3] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise TypeError("Image must be a numpy.ndarray or torch.Tensor.")
    return tensor, orig_shape


def _to_numpy_output(tensor: torch.Tensor, orig_shape: tuple[int, ...]) -> np.ndarray:
    """Chuyển PyTorch NCHW Tensor trên GPU về lại NumPy NHWC array."""
    out = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    if len(orig_shape) == 3:
        out = out[0]
    return out


def _ensure_tensor_grad(grad: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Chuyển đổi gradient về dạng PyTorch Tensor chuẩn NCHW (1, 3, H, W) trên GPU."""
    if isinstance(grad, np.ndarray):
        grad = torch.from_numpy(grad).to(device=device, dtype=torch.float32)
    else:
        grad = grad.to(device=device, dtype=torch.float32)

    # Đưa về dạng NCHW chuẩn
    if grad.ndim == 3:
        if grad.shape[2] == 3:  # (H, W, 3) -> (1, 3, H, W)
            grad = grad.permute(2, 0, 1).unsqueeze(0)
        elif grad.shape[0] == 3:  # (3, H, W) -> (1, 3, H, W)
            grad = grad.unsqueeze(0)
    elif grad.ndim == 4:
        if grad.shape[3] == 3 and grad.shape[1] != 3:  # (1, H, W, 3) -> (1, 3, H, W)
            grad = grad.permute(0, 3, 1, 2)

    return grad


def _initialize_adversarial_image(image: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Khởi tạo nhiễu ngẫu nhiên trong phạm vi L-infinity trực tiếp trên GPU."""
    noise = (torch.rand_like(image) * 2.0 - 1.0) * epsilon
    return torch.clamp(image + noise, 0.0, 1.0)


def _project(image: torch.Tensor, adversarial_image: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Chiếu ảnh đối kháng về phạm vi L-infinity và dải điểm ảnh [0, 1] trên GPU."""
    perturbation = torch.clamp(adversarial_image - image, -epsilon, epsilon)
    return torch.clamp(image + perturbation, 0.0, 1.0)


def tog_vanishing(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Tấn công xóa bỏ đối tượng (Object-Vanishing Attack)."""
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    x, orig_shape = _prepare_tensor(image, victim.device)
    adv_x = _initialize_adversarial_image(x, epsilon)

    for _ in range(num_iterations):
        grad = victim.compute_object_vanishing_gradient(adv_x)
        grad = _ensure_tensor_grad(grad, victim.device)
        adv_x = adv_x - step_size * torch.sign(grad)
        adv_x = _project(x, adv_x, epsilon)

    return _to_numpy_output(adv_x, orig_shape)


def tog_fabrication(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Tấn công tạo đối tượng giả (Object-Fabrication Attack)."""
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)
    x, orig_shape = _prepare_tensor(image, victim.device)
    adv_x = _initialize_adversarial_image(x, epsilon)

    for _ in range(num_iterations):
        grad = victim.compute_object_fabrication_gradient(adv_x)
        grad = _ensure_tensor_grad(grad, victim.device)
        adv_x = adv_x - step_size * torch.sign(grad)
        adv_x = _project(x, adv_x, epsilon)

    return _to_numpy_output(adv_x, orig_shape)


def tog_mislabeling(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    target: Literal["most_likely", "least_likely"],
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Tấn công làm sai lệch nhãn có mục tiêu (Object-Mislabeling Attack)."""
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)

    detections = victim.detect(image)
    if detections is None or len(detections) == 0:
        raise ValueError("No objects detected in the initial image to mislabel.")

    target_detections = generate_attack_targets(
        detections=detections,
        mode=target,
        confidence_threshold=victim.confidence_threshold,
    )

    x, orig_shape = _prepare_tensor(image, victim.device)
    adv_x = _initialize_adversarial_image(x, epsilon)

    for _ in range(num_iterations):
        grad = victim.compute_object_mislabeling_gradient(adv_x, target_detections)
        grad = _ensure_tensor_grad(grad, victim.device)
        adv_x = adv_x - step_size * torch.sign(grad)
        adv_x = _project(x, adv_x, epsilon)

    return _to_numpy_output(adv_x, orig_shape)


def tog_untargeted(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
) -> np.ndarray:
    """Tấn công làm sai lệch nhãn không mục tiêu (Untargeted Attack)."""
    _validate_attack_inputs(image, num_iterations, epsilon, step_size)

    detections = victim.detect(image)
    if len(detections) == 0:
        raise ValueError("No benign objects detected in the image.")

    x, orig_shape = _prepare_tensor(image, victim.device)
    adv_x = _initialize_adversarial_image(x, epsilon)

    for _ in range(num_iterations):
        grad = victim.compute_object_untargeted_gradient(adv_x, detections)
        grad = _ensure_tensor_grad(grad, victim.device)
        adv_x = adv_x - step_size * torch.sign(grad)
        adv_x = _project(x, adv_x, epsilon)

    return _to_numpy_output(adv_x, orig_shape)


class COCOAttackDataset(Dataset):
    def __init__(self, fpaths, image_size):
        self.fpaths = fpaths
        self.image_size = image_size

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        fpath = self.fpaths[idx]
        img, _ = letterbox_image(Image.open(fpath), size=self.image_size)
        return img[0]


def tog_universal(
    victim: YOLOv3TOGModel,
    image: np.ndarray,
    num_iterations: int = 10,
    epsilon: float = 8 / 255.0,
    step_size: float = 2 / 255.0,
    data_path: str = "",
    n_train_samples: int = 100,
    num_workers: int = 4,
) -> np.ndarray:
    """Tạo nhiễu đối kháng toàn cục (Universal Adversarial Perturbation - UAP)."""
    fpaths_train = [os.path.join(data_path, file.name) for file in Path(data_path).iterdir() if file.is_file()]

    if not fpaths_train:
        raise ValueError(f"No images found in path: {data_path}")

    random.shuffle(fpaths_train)
    fpaths_train = fpaths_train[:n_train_samples]

    dataset = COCOAttackDataset(fpaths_train, victim.model_image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    x_ref, orig_shape = _prepare_tensor(image, victim.device)
    noise = torch.zeros_like(x_ref)

    for epoch in range(num_iterations):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_iterations}")

        for batch_img in pbar:
            img_gpu = batch_img.permute(0, 3, 1, 2).to(device=victim.device, dtype=torch.float32)
            adv_img = torch.clamp(img_gpu + noise, 0.0, 1.0)

            grad = victim.compute_object_vanishing_gradient(adv_img)
            grad = _ensure_tensor_grad(grad, victim.device)
            noise = torch.clamp(noise - step_size * torch.sign(grad), -epsilon, epsilon)

    return _to_numpy_output(noise, orig_shape)