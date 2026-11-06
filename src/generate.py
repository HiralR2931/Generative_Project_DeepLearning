"""Stable Diffusion generation helpers (baseline for Milestone 1, reused later)."""
import json
import time
from pathlib import Path

import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)

SCHEDULERS = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "dpm++": DPMSolverMultistepScheduler,
    "pndm": PNDMScheduler,  # SD 1.5 default
}


def load_pipeline(model_id: str, scheduler: str = "dpm++", device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = set_scheduler(pipe, scheduler)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def set_scheduler(pipe, name: str):
    pipe.scheduler = SCHEDULERS[name].from_config(pipe.scheduler.config)
    return pipe


@torch.no_grad()
def empty_prompt_embeds(pipe, n: int):
    """Unconditional ("") embedding used for classifier-free guidance."""
    tok = pipe.tokenizer(
        [""] * n, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
    )
    return pipe.text_encoder(tok.input_ids.to(pipe.device))[0]


def generate(
    pipe,
    prompts=None,
    prompt_embeds=None,
    guidance_scale: float = 7.5,
    num_steps: int = 30,
    seed: int = 42,
    size: int = 512,
):
    """Generate from text prompts OR from pre-computed CLIP hidden states (N, 77, 768).

    A fixed seed makes runs comparable across settings.
    """
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    kwargs = dict(
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        generator=gen,
        height=size,
        width=size,
    )
    if prompt_embeds is not None:
        pe = prompt_embeds.to(pipe.device, dtype=pipe.unet.dtype)
        kwargs["prompt_embeds"] = pe
        kwargs["negative_prompt_embeds"] = empty_prompt_embeds(pipe, len(pe)).to(dtype=pipe.unet.dtype)
    else:
        kwargs["prompt"] = list(prompts)
    t0 = time.time()
    images = pipe(**kwargs).images
    return images, time.time() - t0


def save_run(images, prompts, out_dir: Path, settings: dict, prefix: str = "sample"):
    """Save images + a JSON log of the exact settings (for reproducibility)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (img, p) in enumerate(zip(images, prompts)):
        fn = f"{prefix}_{i:02d}.png"
        img.save(out_dir / fn)
        records.append({"file": fn, "prompt": p, **settings})
    with open(out_dir / f"{prefix}_log.json", "w") as f:
        json.dump(records, f, indent=2)
    return records
