"""Explicit text-to-image denoising loop for Stable Diffusion 1.5 (Milestone 2).

StableDiffusionPipeline hides these steps. Writing them out shows where each part acts:

  1. CLIP text encoder   caption -> (77, 768) token embeddings
  2. Noise schedule      the scheduler picks which timesteps (noise levels) to visit
  3. U-Net               predicts the noise and reads the text through cross-attention
  4. Classifier-free     eps = eps_uncond + w * (eps_text - eps_uncond)
     guidance (CFG)
  5. VAE decoder         4 x 64 x 64 latent -> 512 x 512 RGB image

With the same seed and settings, manual_generate() gives the same image as the pipeline,
which shows the loop below is exactly what the pipeline runs.
"""
from __future__ import annotations

import inspect

import torch
from PIL import Image


@torch.no_grad()
def encode_text(pipe, prompts):
    """Caption(s) -> CLIP ViT-L/14 per-token hidden states, shape (N, 77, 768)."""
    tok = pipe.tokenizer(
        list(prompts),
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return pipe.text_encoder(tok.input_ids.to(pipe.device))[0]


@torch.no_grad()
def manual_generate(pipe, prompt: str, guidance_scale: float = 7.5, num_steps: int = 30,
                    seed: int = 42, size: int = 512, return_timesteps: bool = False):
    """Generate one image with an explicit denoising loop (uses pipe.scheduler as it is set)."""
    device, dtype = pipe.device, pipe.unet.dtype

    # 1. Text -> embeddings. CFG needs two: the prompt and the empty prompt "".
    text_emb = encode_text(pipe, [prompt]).to(dtype)
    use_cfg = guidance_scale > 1.0                     # same rule as the diffusers pipeline
    if use_cfg:
        uncond_emb = encode_text(pipe, [""]).to(dtype)
        context = torch.cat([uncond_emb, text_emb])    # batch of 2: [unconditional, conditional]
    else:
        context = text_emb

    # 2. Noise schedule: the timesteps (noise levels) this sampler will visit.
    sched = pipe.scheduler
    sched.set_timesteps(num_steps, device=device)

    # 3. Start from pure Gaussian noise in latent space (4 x 64 x 64 for a 512 px image).
    gen = torch.Generator(device=device).manual_seed(seed)
    vae_scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)       # 8 for SD 1.5
    shape = (1, pipe.unet.config.in_channels, size // vae_scale, size // vae_scale)
    latents = torch.randn(shape, generator=gen, device=device, dtype=dtype) * sched.init_noise_sigma

    # Stochastic samplers (e.g. DDPM) draw extra noise each step; DDIM takes eta.
    step_args = set(inspect.signature(sched.step).parameters)
    extra = {}
    if "generator" in step_args:
        extra["generator"] = gen
    if "eta" in step_args:
        extra["eta"] = 0.0

    for t in sched.timesteps:
        x = torch.cat([latents] * 2) if use_cfg else latents
        x = sched.scale_model_input(x, t)
        # 4. U-Net predicts the noise; the text enters through cross-attention.
        eps = pipe.unet(x, t, encoder_hidden_states=context).sample
        if use_cfg:
            eps_uncond, eps_text = eps.chunk(2)
            eps = eps_uncond + guidance_scale * (eps_text - eps_uncond)   # classifier-free guidance
        latents = sched.step(eps, t, latents, **extra).prev_sample        # remove a bit of noise

    # 5. VAE decoder: latent -> image in [-1, 1] -> 8-bit RGB.
    img = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    pil = Image.fromarray((img * 255).round().astype("uint8"))
    if return_timesteps:
        return pil, [int(t) for t in sched.timesteps]
    return pil
