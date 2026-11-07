"""Central settings. Change values here instead of inside notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Folders
DATA_DIR = ROOT / "Data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = ROOT / "outputs"
SAMPLES_DIR = OUT_DIR / "samples"

# COCO 2017 validation split
COCO_IMAGES = RAW_DIR / "val2017"
COCO_CAPTIONS = RAW_DIR / "annotations" / "captions_val2017.json"

# Models
CLIP_MODEL = "openai/clip-vit-large-patch14"   # same text encoder as SD 1.5
SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# Preprocessing
IMAGE_SIZE = 512          # SD 1.5 native resolution
MIN_CAPTION_WORDS = 5     # drop very short captions
SUBSET_SIZE = 1000        # fast-experiment subset; full set = 5000

# Generation defaults (baseline)
GUIDANCE_SCALE = 7.5
NUM_STEPS = 30
SCHEDULER = "dpm++"       # one of: ddpm, ddim, dpm++, pndm
SEED = 42


# Handwritten test prompts (second test set, compared against COCO captions)
HANDWRITTEN_PROMPTS = [
    "A brown dog running through a grassy field in sunlight.",
    "Two people riding bicycles along a city street.",
    "A red bus traveling down a road on a cloudy day.",
    "A plate of fresh fruit on a wooden kitchen table.",
    "A small boat floating on a calm lake near mountains.",
]
HANDWRITTEN_SEEDS = [2027, 2028, 2029, 2030, 2031]