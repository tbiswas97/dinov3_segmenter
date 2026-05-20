from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:  # pragma: no cover
    SamAutomaticMaskGenerator = None
    sam_model_registry = None


DEFAULT_REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAM_CHECKPOINT = (
    DEFAULT_REPO_DIR / "pretrained" / "sam" / "sam_vit_b_01ec64.pth"
)
DEFAULT_SAM_MODEL_TYPE = "vit_b"

_SAM_MODEL_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}


def _require_segment_anything():
    if SamAutomaticMaskGenerator is None or sam_model_registry is None:
        raise ImportError(
            "segment_anything is not installed. Install the SAM repo with "
            "`pip install -e /path/to/segment-anything`."
        )


def _default_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def prepare_sam_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        if Image is None:
            raise ImportError("PIL is required to load image paths.")
        image_rgb = np.array(Image.open(image).convert("RGB"))
    else:
        image_rgb = np.asarray(image)

    if image_rgb.ndim != 3:
        raise ValueError(
            f"SAM expects an RGB image with shape H x W x 3, got {image_rgb.shape}"
        )

    if image_rgb.shape[-1] == 4:
        image_rgb = image_rgb[..., :3]
    elif image_rgb.shape[-1] != 3:
        raise ValueError(
            f"SAM expects an RGB image with 3 channels, got {image_rgb.shape}"
        )

    if np.issubdtype(image_rgb.dtype, np.floating):
        if image_rgb.max() <= 1:
            image_rgb = 255 * image_rgb
        image_rgb = np.clip(image_rgb, 0, 255)

    return image_rgb.astype(np.uint8)


def load_sam_model(
    checkpoint_path: str | Path = DEFAULT_SAM_CHECKPOINT,
    model_type: str = DEFAULT_SAM_MODEL_TYPE,
    device: str | torch.device | None = None,
    use_cache: bool = True,
):
    _require_segment_anything()

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    device = _default_device(device)
    cache_key = (model_type, str(checkpoint_path), str(device))

    if not use_cache or cache_key not in _SAM_MODEL_CACHE:
        model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        model.to(device=device)
        model.eval()
        if use_cache:
            _SAM_MODEL_CACHE[cache_key] = model
        else:
            return model

    return _SAM_MODEL_CACHE[cache_key]


def make_sam_mask_generator(
    model,
    *,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    crop_n_layers: int = 0,
    crop_n_points_downscale_factor: int = 1,
    min_mask_region_area: int = 0,
    **kwargs: Any,
):
    _require_segment_anything()
    return SamAutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        **kwargs,
    )


def generate_sam_masks(
    image: str | Path | np.ndarray,
    *,
    mask_generator=None,
    checkpoint_path: str | Path = DEFAULT_SAM_CHECKPOINT,
    model_type: str = DEFAULT_SAM_MODEL_TYPE,
    device: str | torch.device | None = None,
    use_fp16: bool = False,
    **mask_generator_kwargs: Any,
):
    image_rgb = prepare_sam_image(image)

    if mask_generator is None:
        model = load_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            device=device,
        )
        mask_generator = make_sam_mask_generator(model, **mask_generator_kwargs)

    device_type = next(mask_generator.predictor.model.parameters()).device.type
    autocast_enabled = use_fp16 and device_type == "cuda"

    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            return mask_generator.generate(image_rgb)


def postprocess_sam_masks_to_k_labels(
    sam_masks_data: list[dict[str, Any]],
    image_shape_hw: tuple[int, int],
    num_target_labels: int = 64,
    background_label: int = 0,
    min_area_ratio: float = 0.001,
    iou_threshold_for_nms: float = 0.7,
) -> np.ndarray:
    if not sam_masks_data:
        return np.full(image_shape_hw, background_label, dtype=np.uint8)

    height, width = image_shape_hw
    total_pixels = height * width
    min_pixel_area = int(total_pixels * min_area_ratio)

    valid_masks = []
    for mask_data in sam_masks_data:
        area = mask_data["area"]
        if area < min_pixel_area:
            continue

        valid_masks.append(
            {
                "segmentation": mask_data["segmentation"],
                "area": area,
                "score": mask_data.get(
                    "predicted_iou", mask_data.get("stability_score", 0.0)
                ),
                "bbox": mask_data["bbox"],
            }
        )

    valid_masks.sort(key=lambda item: item["score"], reverse=True)

    selected_masks = []
    for current_mask in valid_masks:
        current_binary_mask = current_mask["segmentation"]
        is_redundant = False

        if iou_threshold_for_nms < 1.0:
            for selected_mask in selected_masks:
                selected_binary_mask = selected_mask["segmentation"]
                intersection = np.logical_and(
                    current_binary_mask, selected_binary_mask
                ).sum()
                if intersection == 0:
                    continue

                union = np.logical_or(current_binary_mask, selected_binary_mask).sum()
                if union == 0:
                    continue

                if intersection / union > iou_threshold_for_nms:
                    is_redundant = True
                    break

        if not is_redundant:
            selected_masks.append(current_mask)

    num_foreground_labels = num_target_labels
    if background_label == 0:
        num_foreground_labels -= 1
    num_foreground_labels = min(len(selected_masks), num_foreground_labels)

    label_map = np.full(image_shape_hw, background_label, dtype=np.uint8)
    current_label = background_label + 1 if background_label == 0 else 0

    for mask_info in selected_masks[:num_foreground_labels]:
        label_map[mask_info["segmentation"]] = current_label
        current_label += 1
        if current_label >= np.iinfo(label_map.dtype).max:
            break

    return label_map


def sam_k_segmentation_map(
    image: str | Path | np.ndarray,
    *,
    K: int = 64,
    checkpoint_path: str | Path = DEFAULT_SAM_CHECKPOINT,
    model_type: str = DEFAULT_SAM_MODEL_TYPE,
    device: str | torch.device | None = None,
    mask_generator=None,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    crop_n_layers: int = 0,
    crop_n_points_downscale_factor: int = 1,
    min_mask_region_area: int = 0,
    background_label: int = 0,
    min_area_ratio: float = 0.0005,
    iou_threshold_for_nms: float = 0.7,
    use_fp16: bool = False,
    return_raw_masks: bool = False,
):
    image_rgb = prepare_sam_image(image)

    raw_masks = generate_sam_masks(
        image_rgb,
        mask_generator=mask_generator,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
        use_fp16=use_fp16,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )

    label_map = postprocess_sam_masks_to_k_labels(
        raw_masks,
        image_shape_hw=image_rgb.shape[:2],
        num_target_labels=K,
        background_label=background_label,
        min_area_ratio=min_area_ratio,
        iou_threshold_for_nms=iou_threshold_for_nms,
    )

    if return_raw_masks:
        return label_map, raw_masks

    return label_map
