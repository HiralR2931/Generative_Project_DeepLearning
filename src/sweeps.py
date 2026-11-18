"""Milestone 2: guidance-scale and noise-schedule sweeps, generation log, grids, CLIP score.

SweepRunner generates the same fixed prompts (one fixed seed per prompt) under different
settings, so any difference between images comes from the setting alone. Every image is
saved and logged (one row per image), which gives the generation log for the report.
"""
from __future__ import annotations

import json
import platform
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import config as C
from .generate import SCHEDULERS, generate


# ----------------------------------------------------------------------------- prompts
def m2_prompts(samples_dir=None) -> pd.DataFrame:
    """10 fixed prompts: the 5 COCO captions from the M1 baseline + the 5 handwritten prompts.

    COCO prompts use seeds 42-46; handwritten prompts keep their M1 seeds (2027-2031).
    Images match M1 only on the same device: CPU and CUDA generators draw different noise.
    """
    samples_dir = Path(samples_dir or C.SAMPLES_DIR)
    with open(samples_dir / "baseline_log.json") as f:
        coco = [r["prompt"] for r in json.load(f)]
    rows = [dict(prompt_id=f"coco_{i}", prompt_type="coco", prompt=p, seed=C.SEED + i)
            for i, p in enumerate(coco)]
    rows += [dict(prompt_id=f"hw_{i}", prompt_type="handwritten", prompt=p, seed=s)
             for i, (p, s) in enumerate(zip(C.HANDWRITTEN_PROMPTS, C.HANDWRITTEN_SEEDS))]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- schedulers
def original_scheduler_config(model_id: str = C.SD_MODEL):
    """SD 1.5's own scheduler config (scaled-linear betas, 1000 training steps)."""
    return SCHEDULERS["pndm"].from_pretrained(model_id, subfolder="scheduler").config


def make_scheduler(name: str, base_config, **overrides):
    """Fresh scheduler built from the original config, so settings never leak between runs."""
    return SCHEDULERS[name].from_config(base_config, **overrides)


def visited_timesteps(name: str, base_config, num_steps: int, timestep_spacing: str):
    """Which of the 1000 training timesteps a sampler visits for a given spacing."""
    s = make_scheduler(name, base_config, timestep_spacing=timestep_spacing)
    s.set_timesteps(num_steps)
    return [int(t) for t in s.timesteps]


# ----------------------------------------------------------------------------- runner
class SweepRunner:
    """Runs one setting at a time over all prompts and keeps the generation log."""

    def __init__(self, pipe, prompts: pd.DataFrame, out_dir, model_id: str = C.SD_MODEL,
                 size: int = C.IMAGE_SIZE):
        self.pipe = pipe
        self.prompts = prompts.reset_index(drop=True)
        self.size = size
        self.out_dir = Path(out_dir)
        (self.out_dir / "images").mkdir(parents=True, exist_ok=True)
        self.base_config = original_scheduler_config(model_id)
        # DPM++'s own default spacing; used for every run unless a sweep changes it.
        self.default_spacing = make_scheduler("dpm++", self.base_config).config.timestep_spacing
        self.log = []          # generation log: one dict per image
        self._done = {}        # (setting key, prompt_id) -> (file, seconds, blank): reuse identical runs
        self._warm = False

    def _warmup(self):
        """One short untimed run so CUDA start-up cost is not counted in the first image."""
        generate(self.pipe, prompts=["warm-up"], num_steps=2, seed=0, size=self.size)
        self._warm = True

    def run(self, experiment: str, label: str, scheduler: str = "dpm++",
            guidance_scale: float = C.GUIDANCE_SCALE, num_steps: int = C.NUM_STEPS,
            timestep_spacing: str = None):
        """Generate every prompt with one setting. Identical settings are reused, not re-generated."""
        spacing = timestep_spacing or self.default_spacing
        self.pipe.scheduler = make_scheduler(scheduler, self.base_config, timestep_spacing=spacing)
        key = f"{scheduler}_{spacing}_w{guidance_scale:g}_n{num_steps}"
        if not self._warm:
            self._warmup()

        for row in self.prompts.itertuples():
            reused = (key, row.prompt_id) in self._done
            if reused:
                file, secs, blank = self._done[(key, row.prompt_id)]
            else:
                images, secs = generate(self.pipe, prompts=[row.prompt], guidance_scale=guidance_scale,
                                        num_steps=num_steps, seed=int(row.seed), size=self.size)
                img = images[0]
                file = f"images/{key}__{row.prompt_id}.jpg"
                img.save(self.out_dir / file, quality=92)
                blank = bool(np.asarray(img).max() < 10)   # safety checker returns a black image
                self._done[(key, row.prompt_id)] = (file, secs, blank)
            self.log.append(dict(
                experiment=experiment, setting=label, prompt_id=row.prompt_id,
                prompt_type=row.prompt_type, prompt=row.prompt, seed=int(row.seed),
                scheduler=scheduler, timestep_spacing=spacing, guidance_scale=guidance_scale,
                num_steps=num_steps, seconds=round(secs, 2), reused=reused, blank=blank, file=file,
            ))
        self.save_log()
        new = sum(not r["reused"] for r in self.log if r["experiment"] == experiment and r["setting"] == label)
        print(f"[{experiment}] {label}: {new} new images, {len(self.prompts) - new} reused")

    @property
    def log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    def save_log(self, df: pd.DataFrame = None):
        (df if df is not None else self.log_df).to_csv(self.out_dir / "generation_log.csv", index=False)


def run_info(pipe, extra: dict = None) -> dict:
    """Environment + fixed settings, saved next to the generation log."""
    import diffusers
    import transformers
    info = dict(
        date=time.strftime("%Y-%m-%d %H:%M"),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        dtype=str(pipe.unet.dtype), python=platform.python_version(), torch=torch.__version__,
        diffusers=diffusers.__version__, transformers=transformers.__version__,
        model=C.SD_MODEL, text_encoder=C.CLIP_MODEL, safety_checker="on",
    )
    info.update(extra or {})
    return info


# ----------------------------------------------------------------------------- evaluation
def add_clip_scores(log_df: pd.DataFrame, out_dir, encoder) -> pd.DataFrame:
    """CLIP score = 100 x cosine(image, prompt), computed once per unique image."""
    out_dir = Path(out_dir)
    uniq = log_df.drop_duplicates("file")
    images = [Image.open(out_dir / f).convert("RGB") for f in uniq.file]
    _, text_feats = encoder.encode_text(uniq.prompt.tolist(), return_hidden=False)
    image_feats = encoder.encode_images(images)
    scores = dict(zip(uniq.file, (100 * (text_feats * image_feats).sum(1)).tolist()))
    df = log_df.copy()
    df["clip_score"] = df.file.map(scores).round(2)
    return df


def summarize(log_df: pd.DataFrame) -> pd.DataFrame:
    """One row per setting: images, seconds per image, mean CLIP score (all / COCO / handwritten)."""
    keys = ["experiment", "setting"]
    out = log_df.groupby(keys, sort=False).agg(
        images=("file", "count"), sec_per_image=("seconds", "mean"), clip_mean=("clip_score", "mean"),
        clip_std=("clip_score", "std"), blank_images=("blank", "sum"),
    )
    by_type = log_df.pivot_table(index=keys, columns="prompt_type", values="clip_score", aggfunc="mean")
    by_type.columns = [f"clip_{c}" for c in by_type.columns]
    return out.join(by_type).round(2).reset_index()


# ----------------------------------------------------------------------------- figures
def _short(text: str, width: int = 28, lines: int = 3) -> str:
    return "\n".join(textwrap.wrap(text, width)[:lines])


def make_grid(log_df: pd.DataFrame, out_dir, experiment: str, prompt_ids=None, title: str = None,
              save_as=None, thumb: int = 256):
    """Rows = prompts, columns = settings of one experiment (same seed in every row)."""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    df = log_df[log_df.experiment == experiment]
    settings = list(dict.fromkeys(df.setting))           # keep run order
    pids = prompt_ids or list(dict.fromkeys(df.prompt_id))
    fig, axes = plt.subplots(len(pids), len(settings), squeeze=False,
                             figsize=(2.1 * len(settings) + 1.6, 2.25 * len(pids)))
    for r, pid in enumerate(pids):
        for c, s in enumerate(settings):
            row = df[(df.prompt_id == pid) & (df.setting == s)].iloc[0]
            ax = axes[r][c]
            ax.imshow(Image.open(out_dir / row.file).resize((thumb, thumb)))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(s, fontsize=11)
            if c == 0:
                ax.set_ylabel(_short(row.prompt), fontsize=7.5, rotation=0, ha="right", va="center")
            if "clip_score" in df and pd.notna(row.get("clip_score")):
                ax.set_xlabel(f"CLIP {row.clip_score:.1f}", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=13)
    # keep ~0.6 inch free at the top so the title never overlaps the column labels
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()) if title else None)
    if save_as:
        fig.savefig(save_as, dpi=110, bbox_inches="tight")
    return fig


def make_gallery(log_df: pd.DataFrame, out_dir, experiment: str = "baseline", setting: str = None,
                 ncols: int = 5, save_as=None, thumb: int = 320):
    """Early-samples figure: each generated image with its full prompt (and CLIP score)."""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    df = log_df[log_df.experiment == experiment]
    df = df[df.setting == (setting or df.setting.iloc[0])]
    nrows = int(np.ceil(len(df) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(3.3 * ncols, 3.9 * nrows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, df.itertuples()):
        ax.imshow(Image.open(out_dir / row.file).resize((thumb, thumb)))
        score = f"\nCLIP {row.clip_score:.1f}" if "clip_score" in df and pd.notna(row.clip_score) else ""
        ax.set_title(f"[{row.prompt_type}] " + _short(row.prompt, 40, 3) + score, fontsize=8.5)
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=110, bbox_inches="tight")
    return fig


def plot_guidance_curve(log_df: pd.DataFrame, save_as=None):
    """Mean CLIP score vs guidance scale, for all prompts and per prompt type."""
    import matplotlib.pyplot as plt
    df = log_df[log_df.experiment == "guidance"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, part in [("all prompts", df)] + list(df.groupby("prompt_type")):
        m = part.groupby("guidance_scale").clip_score.mean()
        ax.plot(m.index, m.values, marker="o", label=name, lw=2.2 if name == "all prompts" else 1.4)
    ax.set_xlabel("guidance scale w"); ax.set_ylabel("mean CLIP score")
    ax.set_title("Prompt alignment vs classifier-free guidance"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=130, bbox_inches="tight")
    return fig


def plot_setting_bars(log_df: pd.DataFrame, experiment: str, title: str = None, save_as=None):
    """Mean CLIP score (± std) and seconds per image for each setting of one experiment."""
    import matplotlib.pyplot as plt
    df = log_df[log_df.experiment == experiment]
    order = list(dict.fromkeys(df.setting))
    g = df.groupby("setting").agg(score=("clip_score", "mean"), std=("clip_score", "std"),
                                  sec=("seconds", "mean")).loc[order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.bar(order, g["score"], yerr=g["std"], capsize=4, color="#4C72B0")
    a1.set_ylabel("mean CLIP score"); a1.set_ylim(max(0, g["score"].min() - 6), g["score"].max() + 4)
    a2.bar(order, g["sec"], color="#DD8452"); a2.set_ylabel("seconds per image")
    for a in (a1, a2):
        a.grid(axis="y", alpha=0.3); a.tick_params(axis="x", rotation=20)
    fig.suptitle(title or experiment); fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=130, bbox_inches="tight")
    return fig


def plot_noise_schedule(base_config, num_steps: int = 30,
                        spacings=("leading", "linspace", "trailing"), save_as=None):
    """Signal level sqrt(alpha_bar_t) over the 1000 training steps, and the steps each spacing visits."""
    import matplotlib.pyplot as plt
    s = make_scheduler("dpm++", base_config)
    abar = s.alphas_cumprod.numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.plot(np.sqrt(abar), label="signal  sqrt(alpha_bar_t)")
    a1.plot(np.sqrt(1 - abar), label="noise  sqrt(1 - alpha_bar_t)")
    a1.set_xlabel("training timestep t"); a1.set_title(f"SD 1.5 noise schedule ({base_config.beta_schedule})")
    a1.legend(); a1.grid(alpha=0.3)
    for i, sp in enumerate(spacings):
        ts = visited_timesteps("dpm++", base_config, num_steps, sp)
        a2.scatter(ts, [i] * len(ts), s=14)
        a2.text(1045, i, f"first t = {ts[0]}\nlast t = {ts[-1]}", va="center", fontsize=8)
    a2.set_yticks(range(len(spacings))); a2.set_yticklabels(spacings)
    a2.set_xlim(-20, 1250); a2.set_xlabel("timestep visited")
    a2.set_title(f"timesteps visited with {num_steps} steps"); a2.grid(alpha=0.3)
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=130, bbox_inches="tight")
    return fig
