"""Text -> CLIP embedding pipeline (and image features for alignment checks)."""
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipEncoder:
    """Wraps CLIP ViT-L/14.

    encode_text returns two things:
      * hidden  (N, 77, 768): per-token states -> what Stable Diffusion's U-Net
                cross-attends to (identical to SD 1.5's own text encoder output)
      * pooled  (N, 768):     normalised projected embedding -> for CLIP score / retrieval
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = None):
        self.device = device or get_device()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.max_len = self.tokenizer.model_max_length  # 77

    @torch.no_grad()
    def encode_text(self, captions, batch_size: int = 64, return_hidden: bool = True):
        hidden, pooled = [], []
        for i in range(0, len(captions), batch_size):
            tok = self.tokenizer(
                list(captions[i : i + batch_size]),
                padding="max_length",
                max_length=self.max_len,
                truncation=True,
                return_tensors="pt",
            )
            # input_ids only (no attention mask) -> matches how SD 1.5 encodes prompts
            out = self.model.text_model(input_ids=tok.input_ids.to(self.device))
            if return_hidden:
                hidden.append(out.last_hidden_state.half().cpu())
            feats = self.model.text_projection(out.pooler_output)
            pooled.append(F.normalize(feats, dim=-1).float().cpu())
        hidden = torch.cat(hidden) if return_hidden else None
        return hidden, torch.cat(pooled)

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 32):
        """images: list of PIL images -> normalised (N, 768) features."""
        feats = []
        for i in range(0, len(images), batch_size):
            px = self.processor(images=images[i : i + batch_size], return_tensors="pt").pixel_values
            vision_out = self.model.vision_model(pixel_values=px.to(self.device))
            f = self.model.visual_projection(vision_out.pooler_output)
            feats.append(F.normalize(f, dim=-1).float().cpu())
        return torch.cat(feats)

    def count_truncated(self, captions) -> int:
        """How many captions exceed CLIP's 77-token limit."""
        lengths = [len(self.tokenizer(c).input_ids) for c in captions]
        return sum(l > self.max_len for l in lengths)


def retrieval_recall(image_feats, text_feats, ks=(1, 5, 10)) -> dict:
    """Text->image recall@k, assuming row i of both tensors is a matching pair."""
    sims = text_feats @ image_feats.T                       # (N_text, N_img)
    ranks = sims.argsort(dim=1, descending=True)
    target = torch.arange(len(text_feats)).unsqueeze(1)
    hit_pos = (ranks == target).float().argmax(dim=1)      # rank of the true image
    return {f"R@{k}": (hit_pos < k).float().mean().item() for k in ks}
