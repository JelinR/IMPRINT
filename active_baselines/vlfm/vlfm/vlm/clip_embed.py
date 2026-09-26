# unified_embedder.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
import numpy as np

# CLIP (OpenCLIP)
import open_clip

# SigLIP (Transformers)
from transformers import AutoProcessor, AutoModel


Backend = Literal["clip", "siglip"]
DType = Literal["float32", "float16", "bfloat16"]

MODEL_INFO = {
    "clip" : {
        # 'model_name' : "ViT-B-32",
        # 'pretrained' : 'openai',

        # 'model_name': "ViT-L-14",
        # 'pretrained' : "laion2b_s32b_b82k"

        # 'model_name' : "ViT-L-14-336",
        # 'pretrained' : 'openai',

        # 'model_name' : "ViT-H-14",
        # 'pretrained' : "laion2b_s32b_b79k"

        'model_name' : "EVA02-L-14-336",
        'pretrained' : "merged2b_s6b_b61k",
    },
    "siglip" : {
        # 'model_name' : "google/siglip-base-patch16-224",
        'model_name' : 'google/siglip-base-patch16-256',
        'pretrained' : None,
    }
}


@dataclass
class EncodeOutput:
    """
    Return type for encode_both().
    """
    image_embeds: torch.Tensor   # [B_img, D]
    text_embeds: torch.Tensor    # [B_txt, D]


class CLIP_UnifiedEmbedder:
    """
    Unified wrapper over CLIP (open_clip) and SigLIP (Transformers).

    Features:
    - load('clip', model_name, pretrained)  -> open_clip
    - load('siglip', ckpt_name)             -> HF transformers
    - encode_image(images)  -> [B, D] unit-norm
    - encode_text(texts)    -> [B, D] unit-norm
    - encode_both(images, texts) -> EncodeOutput with both, unit-norm

    Notes on model names:
    - CLIP (open_clip):
        Common combos:
          ("ViT-B-32",  "openai")                # D=512
          ("ViT-B-16",  "openai")                # D=512
          ("ViT-L-14",  "openai")                # D=768
          ("ViT-L-14",  "laion2b_s32b_b82k")     # D=768
          ("ViT-H-14",  "laion2b_s32b_b79k")     # D=1024
          ("EVA02-L-14", "laion2b_s9b_b144k")    # D=1024

    - SigLIP (Transformers):
        Checkpoints (ckpt_name):
          "google/siglip-base-patch16-224"       # D=768
          "google/siglip-base-patch16-256"       # D=768
          "google/siglip-large-patch16-384"      # D=1024
          "google/siglip-so400m-patch14-384"     # D=1152 (large, VRAM heavy)
    """

    def __init__(
        self,
        backend: Backend,
        # model_name: str,
        # pretrained: Optional[str] = None,
        *,
        device: Optional[str] = None,
        dtype: DType = "float32",
    ) -> None:
        self.backend: Backend = backend
        self.model_name = MODEL_INFO[backend]['model_name']
        self.pretrained = MODEL_INFO[backend]['pretrained']
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # dtype handling
        if dtype == "float16":
            self.torch_dtype = torch.float16
        elif dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        # Internal handles
        self._model = None
        self._preprocess = None           # for CLIP images
        self._tokenizer = None            # for CLIP texts
        self._hf_processor = None         # for SigLIP
        self._embed_dim: Optional[int] = None

        self._load()

    # --------------------------- public API ---------------------------

    @torch.inference_mode()
    def get_embed(self, image: np.ndarray = None, txt: str = None, debug: bool = False):
        

        if (image is not None) and (txt is not None):
            print(f"Embed Mode: Multi")
            image_PIL = Image.fromarray(image).convert('RGB')
            embed = self.encode_both(image_PIL, txt)

        elif (image is not None):
            print(f"Embed Mode: Image")
            image_PIL = Image.fromarray(image).convert('RGB')
            embed = self.encode_image(image_PIL)
        
        else:
            print(f"Embed Mode: Text")
            embed = self.encode_text(txt)

        return embed

    @torch.inference_mode()
    def encode_image(
        self,
        images: Union[Image.Image, str, Sequence[Union[Image.Image, str]]],
        *,
        normalize: bool = True,
        batch_size: Optional[int] = None,  # not used for CLIP preprocess; kept for extensibility
    ) -> torch.Tensor:
        """
        Encode one or many images to embeddings [B, D].
        Accepts PIL Image(s) or file path(s).
        """
        pil_list = self._to_pil_list(images)
        if self.backend == "clip":
            assert self._preprocess is not None
            tensor = torch.stack([self._preprocess(img) for img in pil_list], dim=0).to(self.device)
            tensor = tensor.to(self.torch_dtype)
            feats = self._model.encode_image(tensor)
        else:  # siglip
            assert self._hf_processor is not None
            inputs = self._hf_processor(images=pil_list, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self._siglip_image_features(inputs)

        if normalize:
            print(f"Normalizing...")
            feats = torch.nn.functional.normalize(feats, dim=-1)
        else:
            print(f"Not normalizing.")
        return feats

    @torch.inference_mode()
    def encode_text(
        self,
        texts: Union[str, Sequence[str]],
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode one or many texts to embeddings [B, D].
        """
        text_list = texts if isinstance(texts, (list, tuple)) else [texts]

        if self.backend == "clip":
            toks = self._tokenize_clip(text_list)
            feats = self._model.encode_text(toks)
        else:  # siglip
            assert self._hf_processor is not None
            inputs = self._hf_processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self._siglip_text_features(inputs)

        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

    @torch.inference_mode()
    def encode_both(
        self,
        images: Union[Image.Image, str, Sequence[Union[Image.Image, str]]],
        texts: Union[str, Sequence[str]],
        *,
        normalize: bool = True,
    ) -> EncodeOutput:
        """
        Convenience: encode image(s) and text(s) in one call.
        """
        img_emb = self.encode_image(images, normalize=normalize)
        txt_emb = self.encode_text(texts, normalize=normalize)
        return EncodeOutput(image_embeds=img_emb, text_embeds=txt_emb)

    @property
    def embed_dim(self) -> int:
        """Return embedding dimensionality D."""
        if self._embed_dim is None:
            raise RuntimeError("Model not initialized correctly.")
        return self._embed_dim

    # --------------------------- internals ---------------------------

    def _load(self) -> None:
        if self.backend == "clip":
            print(f"Loading CLIP...")
            self._load_open_clip()
        elif self.backend == "siglip":
            print(f"Loading SigLIP...")
            self._load_siglip()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # move dtype (where supported)
        if self.backend == "clip":
            # open_clip supports .to(dtype)
            self._model = self._model.to(self.device, dtype=self.torch_dtype)
        else:
            # HF model .to(device, dtype=...)
            self._model = self._model.to(self.device, dtype=self.torch_dtype)

        self._model.eval()

    def _load_open_clip(self) -> None:
        """
        Load CLIP via open_clip with (model_name, pretrained).
        """
        if self.pretrained is None:
            raise ValueError("For backend='clip', provide 'pretrained' (e.g., 'openai' or a LAION tag).")

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer

        # Discover dimensionality
        self._embed_dim = int(getattr(model, "embed_dim", getattr(model, "text_projection", torch.empty(1)).shape[-1]))

    def _load_siglip(self) -> None:
        """
        Load SigLIP via Transformers AutoModel/AutoProcessor.
        """
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=False)
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=False)

        self._hf_processor = processor
        self._model = model

        # Try to infer embedding dim from config or a dummy forward
        d = getattr(model.config, "projection_dim", None) or getattr(model.config, "hidden_size", None)
        if d is None:
            # Fallback: run a tiny forward to inspect
            with torch.inference_mode():
                dummy = processor(text=["x"], images=[Image.new("RGB", (224, 224), (0, 0, 0))], return_tensors="pt")
                dummy_dict = {k: v for k, v in dummy.items()}
                out = model(**{k: v for k, v in dummy.items()})
                d = int(out.image_embeds.shape[-1])
        self._embed_dim = int(d)

    def _tokenize_clip(self, texts: Sequence[str]) -> torch.Tensor:
        tok = self._tokenizer(texts)
        if isinstance(tok, dict):
            tok = tok["input_ids"]
        return tok.to(self.device)

    # --- SigLIP feature helpers ---

    def _siglip_image_features(self, inputs: dict) -> torch.Tensor:
        """
        Use SigLIP's image pathway.
        Prefer model.get_image_features if available; otherwise, forward and read image_embeds.
        """
        model = self._model
        # Some transformer versions expose .get_image_features()
        if hasattr(model, "get_image_features"):
            feats = model.get_image_features(**inputs)
        else:
            out = model(**inputs)
            if hasattr(out, "image_embeds"):
                feats = out.image_embeds
            else:
                # final hidden state -> project, as a very conservative fallback
                feats = out.vision_model_output.pooler_output  # may differ across versions
        return feats

    def _siglip_text_features(self, inputs: dict) -> torch.Tensor:
        """
        Use SigLIP's text pathway.
        Prefer model.get_text_features if available; otherwise, forward and read text_embeds.
        """
        model = self._model
        if hasattr(model, "get_text_features"):
            feats = model.get_text_features(**inputs)
        else:
            out = model(**inputs)
            if hasattr(out, "text_embeds"):
                feats = out.text_embeds
            else:
                # CLS (index 0) from last_hidden_state as fallback, then project if available
                last = out.text_model_output.last_hidden_state  # [B, T, H]
                feats = last[:, 0, :]                           # [B, H]
                # (Many SigLIP checkpoints include a projection; if missing, cosine still works but dims may differ.)
        return feats

    # --- utilities ---

    @staticmethod
    def _to_pil_list(
        images: Union[Image.Image, str, Sequence[Union[Image.Image, str]]]
    ) -> List[Image.Image]:
        if isinstance(images, (list, tuple)):
            items = images
        else:
            items = [images]
        pil_list: List[Image.Image] = []
        for it in items:
            if isinstance(it, Image.Image):
                pil = it
            elif isinstance(it, str):
                pil = Image.open(it).convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(it)}")
            pil_list.append(pil)
        return pil_list


if __name__ == "__main__":
    import torch

    #pip install "transformers==4.39.3" "tokenizers==0.15.2" "huggingface-hub==0.22.2"

    # ---- CLIP (ViT-L/14 OpenAI) ----
    clip_embedder = CLIP_UnifiedEmbedder(
        backend="clip",
        # model_name="ViT-B-32",
        # pretrained="openai",      # alternatives: "laion2b_s32b_b82k", Vit-L-14
        # dtype="float16",          # fp16 on GPU saves VRAM
    )

    img_e = clip_embedder.encode_image("/mnt/vlfm_query_embed/data/scraped_imgs/ovon_15/backpack/000004.jpg")          # [1, 768]
    txt_e = clip_embedder.encode_text(["a cat", "a dog"])  # [2, 768]
    print(img_e.shape, txt_e.shape)
    print((img_e @ txt_e.T))                               # cosine sims


    # ---- SigLIP (base, 224) ----
    siglip_embedder = CLIP_UnifiedEmbedder(
        backend="siglip",
        # model_name="google/siglip-base-patch16-224",
        # dtype="float16",
    )

    out = siglip_embedder.encode_both(["/mnt/vlfm_query_embed/data/scraped_imgs/ovon_15/backpack/000004.jpg", "/mnt/vlfm_query_embed/data/scraped_imgs/ovon_15/bath cabinet/000007.jpg"], 
                                      ["a cat", "a dog"])
    print(out.image_embeds.shape, out.text_embeds.shape)   # [2, D], [2, D]
    print(torch.argmax(out.image_embeds @ out.text_embeds.T, dim=1))  # retrieval argmax