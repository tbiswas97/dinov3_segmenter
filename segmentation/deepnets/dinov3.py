import numpy as np
import torch
from pathlib import Path

DEFAULT_REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = DEFAULT_REPO_DIR / "pretrained"

MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

MODEL_TO_EMBED_DIM = {
    MODEL_DINOV3_VITS: 384,
    MODEL_DINOV3_VITSP: 384,
    MODEL_DINOV3_VITB: 768,
    MODEL_DINOV3_VITL: 1024,
    MODEL_DINOV3_VITHP: 1280,
    MODEL_DINOV3_VIT7B: 4096,
}

MODEL_NAME = MODEL_DINOV3_VITS
PATCH_SIZE = 16
EMBED_DIM = MODEL_TO_EMBED_DIM[MODEL_NAME]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MODEL_CACHE = {}


def get_n_list(ny, nx, patch_size=PATCH_SIZE):
    if ny % patch_size != 0 or nx % patch_size != 0:
        raise ValueError(
            f"DINOv3 input shape {(ny, nx)} must be divisible by patch size {patch_size}"
        )
    return np.array([(ny // patch_size, nx // patch_size)])


def get_default_weights(model_name=MODEL_NAME, repo_dir=DEFAULT_REPO_DIR):
    weights_dir = Path(repo_dir) / "pretrained"
    weights_subdir = weights_dir / "dinov3"
    compact_name = model_name.replace("dinov3_", "")

    patterns = [
        (weights_subdir, f"{model_name}*.pth"),
        (weights_dir, f"{model_name}*.pth"),
        (weights_subdir, f"dinov3*{compact_name}*.pth"),
        (weights_dir, f"dinov3*{compact_name}*.pth"),
    ]

    candidates = []
    for directory, pattern in patterns:
        if directory.exists():
            candidates.extend(directory.glob(pattern))

    candidates = sorted(set(candidates))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple local weights found for {model_name}: "
            + ", ".join(str(path) for path in candidates)
        )
    return str(candidates[0])


def load_model(
    *,
    repo_dir=DEFAULT_REPO_DIR,
    model_name=MODEL_NAME,
    weights=None,
    pretrained=True,
    device="cpu",
    **hub_kwargs,
):
    if repo_dir is None:
        repo_dir = DEFAULT_REPO_DIR
    repo_dir = Path(repo_dir)

    if weights is None:
        weights = get_default_weights(model_name=model_name, repo_dir=repo_dir)
    elif isinstance(weights, Path):
        weights = str(weights)

    cache_key = (
        str(repo_dir),
        model_name,
        str(weights),
        pretrained,
        str(device),
        tuple(sorted(hub_kwargs.items())),
    )

    if cache_key not in _MODEL_CACHE:
        load_kwargs = {"source": "local", "pretrained": pretrained}
        if weights is not None:
            load_kwargs["weights"] = weights
        load_kwargs.update(hub_kwargs)

        model = torch.hub.load(repo_dir, model_name, **load_kwargs)
        _MODEL_CACHE[cache_key] = model.to(device).eval()

    return _MODEL_CACHE[cache_key]


def extract_features(model, im_torch, *, norm=True):
    device = next(model.parameters()).device
    x = im_torch.to(device)

    if norm:
        mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean) / std

    # dinov3 api n refers to last `n' layers
    with torch.inference_mode():
        feats = model.get_intermediate_layers(
            x,
            n=1,
            reshape=True,
            norm=True,
        )

    return np.array([feats[0].detach().cpu().numpy()[0]], dtype=object)


def build(
    ny,
    nx,
    *,
    device="cpu",
    repo_dir=DEFAULT_REPO_DIR,
    weights=None,
    model_name=MODEL_NAME,
    pretrained=True,
    patch_size=PATCH_SIZE,
    embed_dim=None,
    **hub_kwargs,
):
    if model_name not in MODEL_TO_NUM_LAYERS:
        raise ValueError(f"Unsupported DINOv3 model: {model_name}")

    if embed_dim is None:
        embed_dim = MODEL_TO_EMBED_DIM[model_name]

    model = load_model(
        repo_dir=repo_dir,
        model_name=model_name,
        weights=weights,
        pretrained=pretrained,
        device=device,
        **hub_kwargs,
    )

    return {
        "model": model,
        "N_list": get_n_list(ny, nx, patch_size=patch_size),
        "d_list": np.array([embed_dim]),
        "neigh_size_list": 1.0 * np.array([3]),
        "default_layer": 1,
        "feature_kind": "dinov3",
        "extract_features": extract_features,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "num_layers": MODEL_TO_NUM_LAYERS[model_name],
    }
