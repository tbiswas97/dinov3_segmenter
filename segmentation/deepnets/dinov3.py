import numpy as np
import torch
from pathlib import Path


DEFAULT_REPO_DIR = Path(__file__).resolve().parents[2]
MODEL_NAME = "dinov3_vits16"
PATCH_SIZE = 16
EMBED_DIM = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MODEL_CACHE = {}


def get_n_list(ny, nx, patch_size=PATCH_SIZE):
    if ny % patch_size != 0 or nx % patch_size != 0:
        raise ValueError(
            f"DINOv3 input shape {(ny, nx)} must be divisible by patch size {patch_size}"
        )
    return np.array([(ny // patch_size, nx // patch_size)])


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
    embed_dim=EMBED_DIM,
    **hub_kwargs,
):
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
    }
