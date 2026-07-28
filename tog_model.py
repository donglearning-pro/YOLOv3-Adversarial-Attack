"""YOLOv3-specific losses and image gradients required by TOG attacks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from yolov3_model import ANCHOR_MASKS, YOLOv3Detector, to_nchw


def box_iou_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Tính toán chỉ số IoU (Intersection over Union) giữa hai tập hợp bounding box dạng (x_center, y_center, width, height)."""
    boxes1 = boxes1.unsqueeze(-2)
    boxes2 = boxes2.unsqueeze(0)
    
    # Chuyển đổi tọa độ từ dạng tâm (center xywh) sang tọa độ góc min/max (x_min, y_min, x_max, y_max)
    boxes1_min = boxes1[..., :2] - boxes1[..., 2:] / 2
    boxes1_max = boxes1[..., :2] + boxes1[..., 2:] / 2
    boxes2_min = boxes2[..., :2] - boxes2[..., 2:] / 2
    boxes2_max = boxes2[..., :2] + boxes2[..., 2:] / 2
    
    # Tính diện tích phần giao nhau (intersection)
    intersection_min = torch.maximum(boxes1_min, boxes2_min)
    intersection_max = torch.minimum(boxes1_max, boxes2_max)
    intersection_size = (intersection_max - intersection_min).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    
    # Tính diện tích từng box và chỉ số IoU = Intersection / Union
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    return intersection / (area1 + area2 - intersection + 1e-6)


def encode_yolo_targets(
    boxes: np.ndarray,
    input_shape: tuple[int, int],
    anchors: np.ndarray,
    num_classes: int,
) -> list[np.ndarray]:
    """Mã hóa bounding box tuyệt đối (xyxy) và ID lớp thành 3 tensor target cho 3 quy mô (scale) đặc trưng của YOLOv3."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 3 or boxes.shape[-1] != 5:
        raise ValueError("boxes must have shape (batch, num_boxes, 5).")
    if boxes.size and ((boxes[..., 4] < 0).any() or (boxes[..., 4] >= num_classes).any()):
        raise ValueError("Every class ID must be within the detector class range.")

    input_shape_array = np.asarray(input_shape, dtype=np.int32)
    normalized_boxes = boxes.copy()
    
    # Tính tâm và kích thước, sau đó chuẩn hóa theo kích thước ảnh đầu vào về khoảng [0, 1]
    box_centers = (normalized_boxes[..., :2] + normalized_boxes[..., 2:4]) / 2
    box_sizes = normalized_boxes[..., 2:4] - normalized_boxes[..., :2]
    normalized_boxes[..., :2] = box_centers / input_shape_array[::-1]
    normalized_boxes[..., 2:4] = box_sizes / input_shape_array[::-1]

    # Kích thước lưới cho 3 tầng đặc trưng (stride 32, 16, 8 tương ứng 13x13, 26x26, 52x52)
    grid_shapes = [input_shape_array // stride for stride in (32, 16, 8)]
    targets = [
        np.zeros(
            (boxes.shape[0], grid_height, grid_width, len(mask), 5 + num_classes),
            dtype=np.float32,
        )
        for (grid_height, grid_width), mask in zip(grid_shapes, ANCHOR_MASKS)
    ]

    anchor_boxes = anchors[None, ...]
    anchor_min = -anchor_boxes / 2
    anchor_max = anchor_boxes / 2
    valid_mask = (box_sizes[..., 0] > 0) & (box_sizes[..., 1] > 0)

    for batch_index in range(boxes.shape[0]):
        valid_indices = np.flatnonzero(valid_mask[batch_index])
        if valid_indices.size == 0:
            continue
        valid_sizes = box_sizes[batch_index, valid_indices, None, :]
        box_min = -valid_sizes / 2
        box_max = valid_sizes / 2
        
        # Tìm anchor khớp nhất dựa trên IoU giữa kích thước box thực tế và kích thước các anchor
        intersection_size = np.maximum(np.minimum(box_max, anchor_max) - np.maximum(box_min, anchor_min), 0)
        intersection = intersection_size[..., 0] * intersection_size[..., 1]
        box_area = valid_sizes[..., 0] * valid_sizes[..., 1]
        anchor_area = anchor_boxes[..., 0] * anchor_boxes[..., 1]
        best_anchors = np.argmax(intersection / (box_area + anchor_area - intersection), axis=-1)

        # Gán nhãn target vào đúng vị trí ô lưới (grid cell) và anchor tương ứng
        for valid_position, anchor_index in enumerate(best_anchors):
            box_index = valid_indices[valid_position]
            for layer_index, anchor_mask in enumerate(ANCHOR_MASKS):
                if anchor_index not in anchor_mask:
                    continue
                grid_height, grid_width = grid_shapes[layer_index]
                grid_x = int(np.clip(np.floor(normalized_boxes[batch_index, box_index, 0] * grid_width), 0, grid_width - 1))
                grid_y = int(np.clip(np.floor(normalized_boxes[batch_index, box_index, 1] * grid_height), 0, grid_height - 1))
                mask_index = anchor_mask.index(int(anchor_index))
                class_id = int(normalized_boxes[batch_index, box_index, 4])
                
                # Lưu tọa độ box chuẩn hóa, score objectness (= 1.0) và one-hot encoding của nhãn lớp
                targets[layer_index][batch_index, grid_y, grid_x, mask_index, :4] = normalized_boxes[
                    batch_index, box_index, :4
                ]
                targets[layer_index][batch_index, grid_y, grid_x, mask_index, 4] = 1.0
                targets[layer_index][batch_index, grid_y, grid_x, mask_index, 5 + class_id] = 1.0
    return targets


class YOLOv3TOGModel:
    """Lớp bao bọc (wrapper) mô hình YOLOv3, cung cấp các hàm tính mất mát (loss) và đạo hàm (gradient) cho tấn công TOG."""

    def __init__(self, detector: YOLOv3Detector):
        """Khởi tạo và truy xuất các thuộc tính cần thiết từ bộ phát hiện YOLOv3."""
        self.detector = detector
        self.device = detector.device
        self.model_image_size = detector.model_image_size
        self.confidence_threshold = detector.confidence_threshold
        self.num_classes = detector.num_classes
        self.anchors = detector.anchors

    def detect(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Thực thi dự đoán (inference) trên ảnh đầu vào."""
        return self.detector.detect(image, **kwargs)

    def _decode_for_loss(
        self,
        prediction: torch.Tensor,
        anchors: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Giải mã dự đoán của mạng (raw logits) thành tọa độ tâm (box_xy) và kích thước (box_wh) đã chuẩn hóa để tính hàm loss."""
        _, num_anchors, grid_height, grid_width, _ = prediction.shape
        anchor_tensor = torch.as_tensor(anchors, dtype=prediction.dtype, device=prediction.device).view(
            1, num_anchors, 1, 1, 2
        )
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_height, device=prediction.device),
            torch.arange(grid_width, device=prediction.device),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, grid_height, grid_width, 2).to(prediction.dtype)
        grid_size = torch.tensor([grid_width, grid_height], device=prediction.device, dtype=prediction.dtype)
        input_size = torch.tensor(
            [self.model_image_size[1], self.model_image_size[0]],
            device=prediction.device,
            dtype=prediction.dtype,
        )
        # Tọa độ xy áp dụng sigmoid và cộng offset ô lưới, chia cho kích thước lưới
        box_xy = (torch.sigmoid(prediction[..., :2]) + grid) / grid_size
        # Kích thước wh tính theo hàm mũ exp scaled với anchor
        box_wh = torch.exp(prediction[..., 2:4].clamp(max=10)) * anchor_tensor / input_size
        return grid, box_xy, box_wh

    def _objectness_loss(self, predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        """Tính tổng hàm mất mát Binary Cross-Entropy của Objectness trên tất cả quy mô (scales) YOLO."""
        loss = predictions[0].new_zeros(())
        for prediction, target in zip(predictions, targets):
            target = target.permute(0, 3, 1, 2, 4).contiguous()
            loss += F.binary_cross_entropy_with_logits(prediction[..., 4:5], target[..., 4:5], reduction="sum")
        return loss

    def _full_yolo_loss(self, predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
        """Tính tổng hàm mất mát YOLOv3 đầy đủ (gồm vị trí box, objectness, và phân loại lớp) cho tấn công TOG."""
        batch_size = float(predictions[0].shape[0])
        total_loss = predictions[0].new_zeros(())

        for prediction, target, anchor_mask in zip(predictions, targets, ANCHOR_MASKS):
            target = target.permute(0, 3, 1, 2, 4).contiguous()
            object_mask = target[..., 4:5]
            true_classes = target[..., 5:]
            grid, predicted_xy, predicted_wh = self._decode_for_loss(prediction, self.anchors[anchor_mask])
            predicted_boxes = torch.cat([predicted_xy, predicted_wh], dim=-1)
            grid_height, grid_width = prediction.shape[2:4]

            raw_true_xy = target[..., :2] * torch.tensor(
                [grid_width, grid_height], device=prediction.device, dtype=prediction.dtype
            ) - grid
            anchor_tensor = torch.as_tensor(
                self.anchors[anchor_mask], dtype=prediction.dtype, device=prediction.device
            ).view(1, len(anchor_mask), 1, 1, 2)
            raw_true_wh = torch.log(
                target[..., 2:4]
                * torch.tensor(
                    [self.model_image_size[1], self.model_image_size[0]],
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                / anchor_tensor
                + 1e-16
            )
            raw_true_wh = torch.where(object_mask.bool(), raw_true_wh, torch.zeros_like(raw_true_wh))
            box_loss_scale = 2.0 - target[..., 2:3] * target[..., 3:4]

            ignore_mask = torch.ones_like(object_mask)
            for batch_index in range(prediction.shape[0]):
                true_boxes = target[batch_index, ..., :4][object_mask[batch_index, ..., 0] > 0.5]
                if true_boxes.numel() == 0:
                    continue
                best_iou = box_iou_xywh(predicted_boxes[batch_index], true_boxes).max(dim=-1).values
                ignore_mask[batch_index, ..., 0] = (best_iou < 0.45).to(prediction.dtype)

            # 1. Mất mát vị trí tâm box (xy)
            xy_loss = object_mask * box_loss_scale * F.binary_cross_entropy_with_logits(
                prediction[..., :2], raw_true_xy, reduction="none"
            )
            # 2. Mất mát kích thước box (wh)
            wh_loss = object_mask * box_loss_scale * 0.5 * (raw_true_wh - prediction[..., 2:4]) ** 2
            # 3. Mất mát độ tin cậy objectness (loại trừ ô nền có IoU > 0.45 bằng ignore_mask)
            confidence_loss = object_mask * F.binary_cross_entropy_with_logits(
                prediction[..., 4:5], object_mask, reduction="none"
            ) + (1 - object_mask) * F.binary_cross_entropy_with_logits(
                prediction[..., 4:5], object_mask, reduction="none"
            ) * ignore_mask
            # 4. Mất mát phân loại lớp đối tượng
            class_loss = object_mask * F.binary_cross_entropy_with_logits(
                prediction[..., 5:], true_classes, reduction="none"
            )

            # Cộng tổng các thành phần loss thu gọn trên grid/anchor và trung bình hóa theo batch_size
            total_loss += (xy_loss.sum() + wh_loss.sum() + confidence_loss.sum() + class_loss.sum()) / batch_size

        return total_loss

    def _targets_from_detections(self, detections: np.ndarray | None) -> list[torch.Tensor]:
        """Chuyển đổi các phát hiện dạng mảng numpy thành danh sách tensor target chuẩn hóa trên thiết bị (GPU/CPU)."""
        if detections is None or np.asarray(detections).size == 0:
            boxes = np.empty((1, 0, 5), dtype=np.float32)
        else:
            detections = np.asarray(detections, dtype=np.float32)
            if detections.ndim != 2 or detections.shape[1] < 6:
                raise ValueError("detections must be a two-dimensional detector output array.")
            boxes = detections[:, [-4, -3, -2, -1, 0]][None, ...]
        encoded = encode_yolo_targets(boxes, self.model_image_size, self.anchors, self.num_classes)
        return [torch.from_numpy(target).to(self.device) for target in encoded]

    def _image_gradient(
        self,
        image: np.ndarray,
        loss_function: Callable[[list[torch.Tensor]], torch.Tensor],
    ) -> np.ndarray:
        """Thực hiện lan truyền ngược (backpropagation) để tính đạo hàm (gradient) của hàm loss theo ảnh đầu vào."""
        input_tensor = to_nchw(image, self.device).detach().requires_grad_(True)
        self.detector.network.zero_grad(set_to_none=True)
        loss = loss_function(self.detector.network(input_tensor))
        loss.backward()
        if input_tensor.grad is None:
            raise RuntimeError("The attack loss did not produce an input gradient.")
        return input_tensor.grad.detach().permute(0, 2, 3, 1).cpu().numpy()

    def compute_object_vanishing_gradient(self, image: np.ndarray) -> np.ndarray:
        """Tính gradient giảm thiểu objectness (mục tiêu target = 0) để làm biến mất vật thể (Object Vanishing)."""
        targets = self._targets_from_detections(None)
        return self._image_gradient(image, lambda predictions: self._objectness_loss(predictions, targets))

    def compute_object_fabrication_gradient(self, image: np.ndarray) -> np.ndarray:
        """Tính gradient tối đa hóa objectness (mục tiêu target = 1 ở nền) để tạo ra đối tượng giả (Object Fabrication)."""
        targets = self._targets_from_detections(None)
        for target in targets:
            target[..., 4] = 1.0
        return self._image_gradient(image, lambda predictions: self._objectness_loss(predictions, targets))

    def compute_object_untargeted_gradient(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """Tính gradient làm tăng tối đa loss YOLO gốc (-loss) để gây nhiễu loạn nhận diện không mục tiêu (Untargeted Attack)."""
        targets = self._targets_from_detections(detections)
        return self._image_gradient(image, lambda predictions: -self._full_yolo_loss(predictions, targets))

    def compute_object_mislabeling_gradient(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """Tính gradient giảm thiểu loss YOLO hướng tới nhãn mục tiêu mới để làm sai lệch phân loại (Object Mislabeling)."""
        targets = self._targets_from_detections(detections)
        return self._image_gradient(image, lambda predictions: self._full_yolo_loss(predictions, targets))