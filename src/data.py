"""Load, clean and preprocess COCO image–caption pairs."""
import json
import random
import re
from pathlib import Path

import pandas as pd
from PIL import Image


# ---------- captions ----------
def load_coco_captions(captions_json: Path) -> pd.DataFrame:
    """One row per caption: image_id, file_name, caption."""
    with open(captions_json) as f:
        data = json.load(f)
    id_to_file = {im["id"]: im["file_name"] for im in data["images"]}
    rows = [
        {"image_id": a["image_id"], "file_name": id_to_file[a["image_id"]], "caption": a["caption"]}
        for a in data["annotations"]
    ]
    return pd.DataFrame(rows)


def clean_caption(text: str) -> str:
    """Strip, collapse whitespace, remove trailing spaces/periods, capitalise first letter."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.rstrip(" .")
    return text[:1].upper() + text[1:] if text else text


def clean_captions(df: pd.DataFrame, min_words: int = 5) -> pd.DataFrame:
    """Clean text, drop short captions and duplicates."""
    df = df.copy()
    df["caption"] = df["caption"].map(clean_caption)
    df["n_words"] = df["caption"].str.split().str.len()
    df = df[df["n_words"] >= min_words]
    df = df.drop_duplicates(subset=["image_id", "caption"])
    return df.reset_index(drop=True)


def build_pairs(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """One row per image: a primary caption (for generation) + all captions (as references)."""
    pairs = (
        df.groupby(["image_id", "file_name"])["caption"]
        .apply(list)
        .reset_index(name="all_captions")
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    rng = random.Random(seed)
    pairs["caption"] = [rng.choice(caps) for caps in pairs["all_captions"]]
    pairs["n_refs"] = pairs["all_captions"].str.len()
    return pairs


def sample_subset(pairs: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Reproducible random subset of n image–caption pairs."""
    n = min(n, len(pairs))
    return pairs.sample(n=n, random_state=seed).sort_values("image_id").reset_index(drop=True)


# ---------- images ----------
def load_image(path: Path) -> Image.Image:
    """Open an image as RGB (some COCO images are grayscale)."""
    return Image.open(path).convert("RGB")


def resize_center_crop(img: Image.Image, size: int = 512) -> Image.Image:
    """Resize the short side to `size`, then center-crop to size x size."""
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess_images(pairs: pd.DataFrame, src_dir: Path, dst_dir: Path, size: int = 512) -> list:
    """Save resized/cropped copies of each image. Returns file names that failed to load."""
    from tqdm.auto import tqdm

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for fn in tqdm(pairs["file_name"], desc="Preprocessing images"):
        out = dst_dir / fn
        if out.exists():
            continue
        try:
            resize_center_crop(load_image(Path(src_dir) / fn), size).save(out, quality=95)
        except Exception:  # corrupt / missing file
            failed.append(fn)
    return failed


def to_model_tensor(img: Image.Image):
    """PIL (0–255) -> torch tensor (3, H, W) in [-1, 1], the range the SD VAE expects."""
    import numpy as np
    import torch

    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)



def verify_images(pairs, image_dir):
    """Return file names that are missing or corrupt (checked with PIL verify)."""
    bad = []
    for fn in pairs["file_name"]:
        try:
            with Image.open(Path(image_dir) / fn) as im:
                im.verify()
        except Exception:
            bad.append(fn)
    return bad