# From Words to Worlds — Text-to-Image Generation with Transformers + Diffusion

IE 7615 Deep Learning for AI · Generative Project

We build a text-to-image pipeline that conditions **Stable Diffusion 1.5** on **CLIP ViT-L/14** text embeddings,
then study how guidance scale, sampling steps and noise schedulers affect image quality (FID, IS) and
prompt alignment (CLIP score). Dataset: **MS-COCO 2017 val** (5,000 images, ~25,000 captions).

```
caption ──► CLIP tokenizer ──► CLIP text encoder ──► (77 × 768) embedding
                                                          │ cross-attention
random noise ──► U-Net denoising (N steps, guidance w) ◄──┘
                         │
                         ▼
                    VAE decoder ──► 512 × 512 image
```

## Repository structure

```
words-to-worlds/
├── notebooks/
│   ├── 01_data_and_embeddings.ipynb   # download, clean, preprocess, CLIP embeddings
│   └── 02_baseline_samples.ipynb      # 5 prompt→image test runs + guidance preview
├── src/
│   ├── config.py        # all paths, model IDs and default settings
│   ├── data.py          # caption cleaning, pairing, image resize/crop
│   ├── embeddings.py    # CLIP text/image encoder, retrieval check
│   └── generate.py      # Stable Diffusion pipeline, schedulers, run logging
├── scripts/
│   └── download_coco.sh # COCO val2017 images + captions
├── outputs/samples/     # generated images + JSON logs (committed)
├── data/                # downloaded + processed data (git-ignored)
└── requirements.txt
```

## Setup

**Google Colab (recommended, free T4 GPU)**
1. Open a notebook from `notebooks/` in Colab and set *Runtime → GPU*.
2. Edit `REPO_URL` in the first cell, then run all cells. The first cell clones the repo and installs requirements.

**Local (NVIDIA GPU, ~8 GB VRAM)**
```bash
git clone https://github.com/YOUR_USERNAME/words-to-worlds.git
cd words-to-worlds
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_coco.sh
jupyter lab
```

## How to run (Milestone 1)

| Step | Notebook | Output |
|---|---|---|
| 1. Data + embeddings | `01_data_and_embeddings.ipynb` | `data/processed/pairs_*.json`, `images_512/`, `clip_text_*.pt` |
| 2. Baseline samples | `02_baseline_samples.ipynb` | `outputs/samples/baseline_*.png`, `baseline_log.json`, `guidance_preview.png` |

Run them in order; notebook 02 reads the embeddings cached by notebook 01. All randomness uses `SEED = 42` (see `src/config.py`).

## Data processing

- **Captions:** whitespace cleaned, captions under 5 words dropped, duplicates removed. One random caption per image is the prompt; all captions are kept as references.
- **Images:** converted to RGB, short side resized to 512, center-cropped to 512×512, normalised to [-1, 1] for the VAE.
- **Embeddings:** CLIP ViT-L/14 per-token hidden states (77×768, fp16) for the 1k subset and pooled 768-d features for all 5k.

## Milestones

- [ ] **M1 — Setup & data (weeks 1–2):** repo, preprocessing, text→embedding notebook, 5 sample runs, proposal
- [ ] **M2 — Model integration (weeks 3–4):** guidance + scheduler + steps sweeps, early samples
- [ ] **M3 — Evaluation (week 5):** FID, IS, CLIP score, sensitivity plots
- [ ] **M4 — Final (by Dec 13):** report, gallery, slides, demo

## Team

| Member | Role |
|---|---|
| [Name] | Data lead |
| [Name] | Model lead |
| [Name] | Evaluation lead |
| [Name] | Report & repo lead |

## Data and model licenses

- COCO annotations: CC BY 4.0; images follow the Flickr terms of use. Data is not redistributed in this repo.
- Stable Diffusion 1.5: CreativeML OpenRAIL-M license. The built-in safety checker is left enabled.
