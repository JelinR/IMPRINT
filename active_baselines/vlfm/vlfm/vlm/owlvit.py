from scipy.special import expit as sigmoid
from time import time
from PIL import Image
import requests
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Union, Optional, Dict, Any
from dataclasses import dataclass

# from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers import OwlViTProcessor, OwlViTForObjectDetection


from transformers import Owlv2Processor, Owlv2ForObjectDetection


from vlfm.vlm.detections import ObjectDetections

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

@dataclass
class Buffer_Out:
    logits = None
    target_pred_boxes = None

class OwlViT_Detector:

    def __init__(self, 
                    model_id: str = "google/owlvit-base-patch32",
                    precsion: str = "auto"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.model     = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device).eval()

        self.query_vector = None

        # Pick precision
        if self.device == "cuda":
            if precision == "fp16" or (precision == "auto" and torch.cuda.get_device_capability()[0] >= 7):
                self.autocast_dtype = torch.float16
                self.model.to(dtype=torch.float16)
            elif precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
                self.autocast_dtype = torch.bfloat16
                self.model.to(dtype=torch.bfloat16)
            else:
                self.autocast_dtype = None
        else:
            self.autocast_dtype = None

    @torch.inference_mode()
    def extract_query_vector(self, texts, source_image=None, with_objectness=False, plot_result=False):
        ### Extracting the Query Vector

        if type(source_image) == np.ndarray: source_image = Image.fromarray(source_image)

        #Extract the text feature
        inputs = self.processor(text=texts, return_tensors="pt").to(self.device)
        text_feature = self.model.owlvit.get_text_features(input_ids = inputs.input_ids,
                                                    attention_mask = inputs.attention_mask)     #Shape: (1, 512)
        text_feature = F.normalize(text_feature, dim=-1)

        if source_image is None:
            print(f"Setting text feature as the Query Vector...")
            self.query_vector = text_feature[0]
            return

        #Convert image to array, and then obtain the patch embeddings : (1, 60, 60, 768)
        source_pixel_values = self.processor(images=source_image, return_tensors="pt").pixel_values.to(self.device)
        source_image_feature_map = self.model.image_embedder(source_pixel_values)[0]

        #Flatten the source image features : (1, 60, 60, 768) -> (1, 60*60, 768)
        #Project to a lower subspace : (1, 3600, 768) -> (1, 3600, 512)
        batch_size, height, width, hidden_size = source_image_feature_map.shape
        source_image_features = source_image_feature_map.reshape(batch_size, height * width, hidden_size)
        source_image_proj_features = self.model.class_predictor(source_image_features)[1]

        #Measures: Objectness, Similarity with Text Query
        source_boxes = None
        if plot_result:
            source_boxes = self.model.box_predictor(source_image_features, feature_map=source_image_feature_map)
            source_boxes = np.array(source_boxes[0].detach())

        if with_objectness:
            objectnesses = self.model.objectness_predictor(source_image_features)[0]
            objectnesses = np.array(objectnesses[0].detach())
            objectnesses = torch.sigmoid(objectnesses) #.squeeze(-1)
        
        source_image_proj_features = F.normalize(source_image_proj_features, dim=-1)

        similarities = torch.einsum("bnd,bd->bn", source_image_proj_features, text_feature)
        similarities = torch.sigmoid(similarities)

        #Get top-5 predictions wrt similarity scores. Then, get the embedding amongst that has the best objectness.
        if with_objectness:
            top_k = 5
            similarity_thresh = np.partition(similarities, -top_k)[-top_k]
            top_k_inds = np.argwhere(similarities >= similarity_thresh)

            top_ind = top_k_inds[ np.argmax(objectnesses[top_k_inds]) ]
            top_embed = source_image_proj_features[0][top_ind]
            
        else:
            top_embed = source_image_proj_features[0][ torch.argmax(similarities) ]
            print(f" Query Similarity : {torch.max(similarities)}")

            
            if plot_result:
                top_k = 5
                similarity_thresh = np.partition(similarities, -top_k)[-top_k]
                top_k_inds = np.argwhere(similarities >= similarity_thresh)[:, 0]

                self.plot_image_with_bbox(source_pixel_values, source_boxes[top_k_inds], similarities[top_k_inds])

        #Assign query vector
        self.query_vector = top_embed

    @torch.inference_mode()
    def is_query_in_image(self, target_image, return_box=False, plot_result=False):

        assert self.query_vector is not None, "Please extract query vector first."

        if type(target_image) == np.ndarray: target_image = Image.fromarray(target_image)

        # Process target image
        target_pixel_values = self.processor(images=target_image, return_tensors="pt").pixel_values.to(self.device)
        target_image_feature_map = self.model.image_embedder(target_pixel_values)[0]

        #Flatten the source image features : (1, 60, 60, 768) -> (1, 60*60, 768)
        batch_size, height, width, hidden_size = target_image_feature_map.shape
        target_image_features = target_image_feature_map.reshape(batch_size, height * width, hidden_size)

        #With the query vector, obtain class prediction for all patches
        target_class_predictions = self.model.class_predictor(target_image_features,
                                                        self.query_vector[None, None, ...])[0]  #Shape: (1, 3600, 1)
        target_logits = target_class_predictions[0]                                             #Shape: (3600, 1)

        if return_box or plot_result:
            target_boxes = self.model.box_predictor(target_image_features, 
                                            feature_map=target_image_feature_map)
            target_boxes = np.array(target_boxes[0].detach().cpu())

            top_ind = torch.argmax(target_logits[:, 0], axis=0)
            top_score = torch.sigmoid(target_logits[top_ind, 0])

            if plot_result: self.plot_image_with_bbox(target_pixel_values.detach().cpu(), 
                                                        [target_boxes[top_ind]], [top_score])
        
        #Obtain the best prediction
        top_ind = torch.argmax(target_logits[:, 0], axis=0)
        top_score = torch.sigmoid(target_logits[top_ind, 0])
        print(f"Query is in Target Image with confidence: {top_score}")

        if return_box: return top_score, target_boxes[top_ind]      #TODO: Only returns top detection. Need to return all detections.
        return top_score

    def conv_to_xyxy_bbox(self, owl_bbox):

        cx, cy, w, h = owl_bbox

        lower_left = (cx - w/2, cy - h/2)
        upper_right = (cx + w/2, cy + h/2)
        return [lower_left[0], lower_left[1], upper_right[0], upper_right[1]]

    def predict(self, image: np.ndarray):
        """"
        Returns detection as an instance of ObjectDetections. Follows VLFM (GroundingDino) compatibility.
        """

        assert self.query_vector is not None, "Please extract the query vector before detection!"

        score, box = self.is_query_in_image(image, return_box = True)   #Returns logit score, and normalized bounding box
        box_xyxy = self.conv_to_xyxy_bbox(box)

        score = score.unsqueeze(0).detach().cpu()               #TODO: Assumes score and detection will be one. Need to account for multiple detections.
        box_xyxy = torch.tensor(box_xyxy).unsqueeze(0)

        detection = ObjectDetections(boxes = box_xyxy, 
                                     logits = score, 
                                     phrases = [str(np.random.randint(0, 500)) for _ in range(len(box_xyxy))],
                                     image_source = image, 
                                     fmt = "xyxy")
        return detection

    def get_preprocessed_image(self, pixel_values):
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    def plot_image_with_bbox(self, image_pixel_values, boxes, similarities):

        assert len(boxes) == len(similarities), "Please provide similarity values corresponding to the boxes. Should be equal length lists."

        # For visualization, we need the preprocessed source image (i.e. padded and resized, but not yet normalized)
        unnormalized_source_image = self.get_preprocessed_image(image_pixel_values)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(unnormalized_source_image, extent=(0, 1, 1, 0))
        ax.set_axis_off()

        for box, similarity in zip(boxes, similarities):

            cx, cy, w, h = box
            ax.plot(
                [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
                color='lime',
            )

            ax.text(
                cx - w / 2 + 0.015,
                cy + h / 2 - 0.015,
                f'Similarity: {float(similarity):1.2f}',
                ha='left',
                va='bottom',
                color='black',
                bbox={
                    'facecolor': 'white',
                    'edgecolor': 'lime',
                    'boxstyle': 'square,pad=.3',
                },
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_title(f'Top {len(boxes)} objects by Similarity')


class OwlViT_Detector_t:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model     = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")        
        self.model.to(self.device).eval()

        torch.backends.cudnn.benchmark = True
        self.query_vector = None

    def extract_query_vector(self, texts, source_image=None, with_objectness=False, plot_result=False):
        
        # Text features (stay on device)
        inputs = self.processor(text=texts, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            text_feature = self.model.owlvit.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )  # shape (B, D) on device

        if source_image is None:
            print(f"Setting text feature as the Query Vector...")
            self.query_vector = text_feature[0]  # keep on device
            return

        # Image → pixel_values (no PIL needed)
        source_pixel_values = self.processor(images=source_image, return_tensors="pt").pixel_values.to(self.device)

        with torch.inference_mode():
            source_image_feat_map = self.model.image_embedder(source_pixel_values)[0]           # (1,H,W,C)
            b,h,w,d = source_image_feat_map.shape
            source_image_feats = source_image_feat_map.reshape(b, h*w, d)                                    # (1,N,D)

            # class_predictor sometimes returns (logits, proj). We need proj (the embedding space compatible with text).
            source_image_proj_feats = self.model.class_predictor(source_image_feats)[1]                         # (1,N,Dproj) on device

            if plot_result:
                boxes = self.model.box_predictor(source_image_proj_feats, feature_map=source_image_feat_map)[0]    # (N,4) torch on device

            if with_objectness:
                obj = self.model.objectness_predictor(source_image_proj_feats)[0]                 # (N,1)
                obj = torch.sigmoid(obj).squeeze(-1)                            # (N,)

            # Similarity (torch only)
            # proj: (1,N,D), text_feature: (1,D) → (1,N)
            sims = torch.matmul(source_image_proj_feats, text_feature.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (N,)
            sims = torch.sigmoid(sims)                                           # (N,)

            if with_objectness:
                topk = min(5, sims.numel())
                sim_vals, idxs = torch.topk(sims, k=topk, dim=0)
                best_idx = idxs[torch.argmax(obj[idxs])]
                top_embed = source_image_proj_feats[0, best_idx]                                    # (D,)
            else:
                best_idx = torch.argmax(sims)
                top_embed = source_image_proj_feats[0, best_idx]
                # only print tiny scalar to CPU to avoid sync:
                print(f" Query Similarity : {sims[best_idx].item():.4f}")

            if plot_result:
                topk = min(5, sims.numel())
                _, top_idxs = torch.topk(sims, k=topk, dim=0)
                # move only what plotting needs to CPU/NumPy
                self.plot_image_with_bbox(
                    source_pixel_values.detach().cpu(),
                    boxes[top_idxs].detach().cpu().numpy(),
                    sims[top_idxs].detach().cpu().numpy(),
                )

        self.query_vector = top_embed  # keep on device

    def is_query_in_image(self, target_image, return_box=False, plot_result=False):
        assert self.query_vector is not None, "Please extract query vector first."

        target_pixel_values = self.processor(images=target_image, return_tensors="pt").pixel_values.to(self.device)

        with torch.inference_mode():
            target_image_feat_map = self.model.image_embedder(target_pixel_values)[0]       # (1,H,W,C)
            
            #Flatten the source image features : (1, 60, 60, 768) -> (1, 60*60, 768)
            b,h,w,d = target_image_feat_map.shape
            target_image_feats = target_image_feat_map.reshape(b, h*w, d)                                # (1,N,D)

            #With the query vector, obtain class prediction for all patches
            target_class_logits = self.model.class_predictor(
                target_image_feats, self.query_vector[None, None, ...]
            )[0][0]                                                   # (N,1) → take [0] → (N,1)
            # target_class_logits = target_class_logits.squeeze(-1).squeeze(1).squeeze(1)                                     # (N,)

            if return_box or plot_result:
                boxes = self.model.box_predictor(target_image_feats, feature_map=target_image_feat_map)[0]  # (N,4)

            top_ind = torch.argmax(target_class_logits[:, 0])
            top_score = torch.sigmoid(target_class_logits[top_ind, 0])

            if plot_result:
                self.plot_image_with_bbox(
                    target_pixel_values.detach().cpu(),
                    boxes[top_ind:top_ind+1].detach().cpu().numpy(),
                    top_score.detach().cpu().view(1).numpy(),
                )

        print(f"Query is in Target Image with confidence: {top_score.item():.4f}")
        if return_box:
            return top_score, boxes[top_ind]  # both torch tensors on device
        return top_score


    def conv_to_xyxy_bbox(self, owl_bbox):

        cx, cy, w, h = owl_bbox

        lower_left = (cx - w/2, cy - h/2)
        upper_right = (cx + w/2, cy + h/2)
        return [lower_left[0], lower_left[1], upper_right[0], upper_right[1]]

    def predict(self, image: np.ndarray):
        score, box = self.is_query_in_image(image, return_box=True)
        score = score.unsqueeze(-1)

        # box is (4,) center-format; convert → xyxy in torch on device
        cx, cy, w, h = box
        box_xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=0).unsqueeze(0)  # (1,4)

        # keep tensors; only small python bits for phrases
        detection = ObjectDetections(
            boxes = box_xyxy,                            # torch
            logits = score,                              # torch scalar
            phrases = [str(np.random.randint(0, 500))],  # tiny python list; fine
            image_source = image,
            fmt = "xyxy"
        )
        return detection

    def get_preprocessed_image(self, pixel_values):
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    def plot_image_with_bbox(self, image_pixel_values, boxes, similarities):

        assert len(boxes) == len(similarities), "Please provide similarity values corresponding to the boxes. Should be equal length lists."

        # For visualization, we need the preprocessed source image (i.e. padded and resized, but not yet normalized)
        unnormalized_source_image = self.get_preprocessed_image(image_pixel_values)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(unnormalized_source_image, extent=(0, 1, 1, 0))
        ax.set_axis_off()

        for box, similarity in zip(boxes, similarities):

            cx, cy, w, h = box
            ax.plot(
                [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
                color='lime',
            )

            ax.text(
                cx - w / 2 + 0.015,
                cy + h / 2 - 0.015,
                f'Similarity: {similarity:1.2f}',
                ha='left',
                va='bottom',
                color='black',
                bbox={
                    'facecolor': 'white',
                    'edgecolor': 'lime',
                    'boxstyle': 'square,pad=.3',
                },
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_title(f'Top {len(boxes)} objects by Similarity')


class Owlv2_Detector_t:

    def __init__(self, 
                 detect_thresh: float = 0.4, 
                 model_id: str = "google/owlv2-base-patch16",   #"google/owlv2-base-patch16-ensemble"
                 precision: str = "auto",
                 device=None,
                    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()

        # Pick precision
        if self.device == "cuda":
            if precision == "fp16" or (precision == "auto" and torch.cuda.get_device_capability()[0] >= 7):
                self.autocast_dtype = torch.float16
                self.model.to(dtype=torch.float16)
            elif precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
                self.autocast_dtype = torch.bfloat16
                self.model.to(dtype=torch.bfloat16)
            else:
                self.autocast_dtype = None
        else:
            self.autocast_dtype = None

        self.query_txt = None
        self.detect_thresh = detect_thresh

    def set_query(self, texts):

        if isinstance(texts, str):
            self.query_txt = [[texts]]
        elif isinstance(texts, list):
            self.query_txt = [texts]
        else:
            raise TypeError("texts must be a str or List[str]")

        print(f"Set text query : {self.query_txt}")

    @torch.inference_mode()
    def is_query_in_image(self, target_image: torch.Tensor, plot_result: bool=False):

        if self.query_txt is None:
            raise RuntimeError("Call set_query(...) first.")

        target_image = self._ensure_pil(target_image)

        inputs = self.processor(text=self.query_txt, images=target_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        #Target image sizes to resclae box preds
        target_sizes = torch.tensor([(target_image.height, target_image.width)], device=self.device)

        #Convert outputs (bbox, logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.detect_thresh,
        )

        result = results[0]
        boxes, scores, text_labels = result["boxes"], result["scores"], result["labels"]

        # Normalize (x1, y1, x2, y2) by width and height
        h, w = target_image.height, target_image.width
        norm_factors = torch.tensor([w, h, w, h], device=boxes.device)
        boxes = boxes / norm_factors

        if len(boxes > 0):
            print(f"Top Detection Score: {max(scores)}")
    
        else:
            print(f"No Detections found under threshold ({self.detect_thresh})")

        if plot_result:
            self.plot_image_with_bbox(target_image, boxes.detach().cpu().numpy(), scores.detach().cpu().numpy())
    
        return scores, boxes

    @torch.inference_mode()
    def predict(self, image: np.ndarray):

        scores, boxes = self.is_query_in_image(image)

        detection = ObjectDetections(
            boxes = boxes.detach().cpu(),                                # torch
            logits = scores.detach().cpu(),                              # torch scalar
            phrases = self.query_txt,
            image_source = image,
            fmt = "xyxy"
        )
        return detection


    # ----- Utils ------


    @staticmethod
    def _ensure_pil(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:

        #If PIL
        if isinstance(img, Image.Image):
            return img.convert("RGB")

        #If Numpy Array
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            return Image.fromarray(img.astype(np.uint8)).convert("RGB")

        #If Torch Tensor
        if isinstance(img, torch.Tensor):
            # Accept CHW [0..1] or [0..255], or HWC
            t = img.detach().cpu()
            if t.ndim == 3:
                if t.shape[0] in (1,3):  # CHW
                    t = t.mul(255.0) if t.max() <= 1.0 else t
                    t = t.byte().clamp(0,255)
                    t = t.permute(1,2,0).numpy()
                else:  # HWC
                    t = t.mul(255.0) if t.max() <= 1.0 else t
                    t = t.byte().clamp(0,255).numpy()
            else:
                raise ValueError("Expected 3D tensor image (CHW or HWC).")
            return Image.fromarray(t).convert("RGB")
        raise TypeError("Unsupported image type. Use PIL.Image, numpy array, or torch.Tensor.")

    def plot_image_with_bbox(self, image_pil, boxes, similarities):

        assert len(boxes) == len(similarities), "Please provide similarity values corresponding to the boxes. Should be equal length lists."

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_axis_off()

        ax.imshow(image_pil)

        # Normalize (x1, y1, x2, y2) by width and height
        h, w = image_pil.height, image_pil.width
        norm_factors = np.array([w, h, w, h])

        for sim, box_xyxy in zip(similarities, boxes):


            box_xyxy = box_xyxy * norm_factors

            x1, y1, x2, y2 = [float(v) for v in box_xyxy]

            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                            fill=False, linewidth=2))

            # add similarity score text above the box
            ax.text(
                x1, y1 - 5, f"{sim:.3f}",
                fontsize=10, color="white", backgroundcolor="red"
            )


        ax.set_title(f'Top {len(boxes)} objects by Similarity')
        plt.show()


class Owlv2_Detector_img_cond:

    def __init__(self, 
                 detect_thresh: float = 0.99, 
                #  model_id: str = "google/owlv2-base-patch16",   
                 model_id: str = "google/owlv2-base-patch16-ensemble",
                 precision: str = "auto",
                 device=None,
                    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()

        # Pick precision
        if self.device == "cuda":
            if precision == "fp16" or (precision == "auto" and torch.cuda.get_device_capability()[0] >= 7):
                self.autocast_dtype = torch.float16
                self.model.to(dtype=torch.float16)
            elif precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
                self.autocast_dtype = torch.bfloat16
                self.model.to(dtype=torch.bfloat16)
            else:
                self.autocast_dtype = None
        else:
            self.autocast_dtype = None

        self.query_txt = None
        self.txt_emb, self.img_emb = None, None
        self.detect_thresh = detect_thresh

    @torch.inference_mode()
    def extract_query_vector(self, texts, source_image, plot_results=False):

        source_image = self._ensure_pil(source_image)

        # ---- Get Image and Text feature embeddings -----

        is_set = self.is_txt_query_set(texts)
        if not is_set:

            #Pre-process image and text
            img_processed = self.processor(images=source_image, return_tensors="pt").pixel_values.to(self.device)
            txt_processed = self.processor(text=self.query_txt, return_tensors="pt")

            self.txt_emb, img_feat_map, _ = self.model.image_text_embedder(
                                                input_ids = txt_processed.input_ids.to(self.device),
                                                pixel_values = img_processed,
                                                attention_mask = txt_processed.attention_mask.to(self.device)
                                            )

            # Rearrange feature map
            b, h, w, d = img_feat_map.shape
            img_feats = img_feat_map.reshape(b, h * w, d)   #Shape : (1, 60*60, 512)

            #Get feature embeddings. Projected into same space as text embeddings (with 512-dim)
            img_embs = self.model.class_predictor(img_feats)[1]      #Shape : (1, 60*60, 512)

        else:
            #Pre-process image and text
            img_processed = self.processor(images=source_image, return_tensors="pt").pixel_values.to(self.device)

            #Get image features
            img_feat_map = self.model.image_embedder(img_processed)[0]     #Shape : (1, 60, 60, 768)

            # Rearrange feature map
            b, h, w, d = img_feat_map.shape
            img_feats = img_feat_map.reshape(b, h * w, d)

            #Get feature embeddings
            img_embs = self.model.class_predictor(img_feats)[1]

        # ----- Get relevant image query embedding -----
        # Get boxes and class embeddings (the latter conditioned on query embedding)
        logits = self.model.class_predictor(
            img_feats,
            self.txt_emb[:, None, :],       #Shape : (batch, 1, 512)
        )[0]                                #Shape : (batch, 60*60, 1)

        self.img_emb = img_embs[0, torch.argmax(logits)]

        if plot_results:

            #Get unnormalized image
            unnorm_img = get_preprocessed_image(img_processed)

            #Get the bounding boxes and logit scores
            boxes = self.model.box_predictor(
                img_feats, feature_map=img_feat_map
            )                                           #Shape : (1, 60*60, 4)

            boxes = np.array(boxes[0].detach())

            # Take the highest scoring logit
            top_ind = torch.argmax(logits).item()
            score = torch.sigmoid(logits[0, top_ind, 0]).item()
            print(f'Source Score : {score}')

            #Plotting
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(unnorm_img, extent=(0, 1, 1, 0))
            ax.set_axis_off()

            cx, cy, w, h = boxes[top_ind]
            ax.plot(
                [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
                color='lime',
            )

            ax.text(
                cx - w / 2 + 0.015,
                cy + h / 2 - 0.015,
                f'Score: {score:1.2f}',
                ha='left',
                va='bottom',
                color='black',
                bbox={
                    'facecolor': 'white',
                    'edgecolor': 'lime',
                    'boxstyle': 'square,pad=.3',
                },
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_title(f'Closest match')

    def is_txt_query_set(self, texts):

        if isinstance(texts, str):
            query_txt = [[texts]]
        elif isinstance(texts, list):
            query_txt = [texts]

        assert (len(query_txt) == 1) and (len(query_txt[0]) == 1)

        if (self.query_txt is not None) and \
            (query_txt[0][0] == self.query_txt[0][0]):
            return True

        self.query_txt = query_txt
        return False

    @torch.inference_mode()
    def is_query_in_image(self, target_image: torch.Tensor, plot_result: bool=False):

        if self.query_txt is None:
            raise RuntimeError("Call extract_query_vector method first.")

        target_image = self._ensure_pil(target_image)

        target_img_processed = self.processor(images=target_image, return_tensors="pt").pixel_values.to(self.device)
        target_img_feat_map = self.model.image_embedder(target_img_processed)[0]                  #Shape : (1, 60, 60, 768)

        # Rearrange feature map
        b, h, w, d = target_img_feat_map.shape
        target_img_feats = target_img_feat_map.reshape(b, h * w, d)

        #Get logits and boxes, with respect to reference image embedding
        target_logits = self.model.class_predictor(target_img_feats,
                                                    self.img_emb[None, None, ...])[0]

        target_boxes = self.model.box_predictor(
            target_img_feats, feature_map=target_img_feat_map
        )

        #Get top box (in xyxy format)
        target_outputs = Buffer_Out()
        target_outputs.logits = target_logits
        target_outputs.target_pred_boxes = target_boxes

        target_outs = self.processor.post_process_image_guided_detection(target_outputs, 
                                                                    nms_threshold = 1.0,
                                                                    threshold = 0.0,
                                                                    target_sizes=torch.tensor([target_image.height, target_image.width]).unsqueeze(0).to(self.device))

        top_ind = torch.argmax(target_outs[0]["scores"])
        top_box = target_outs[0]["boxes"][top_ind]

        #Get top score
        top_ind = torch.argmax(target_logits).item()
        top_score = torch.sigmoid(target_logits[0, top_ind, 0])

        if top_score.item() < self.detect_thresh:
            print(f"No Detections found under threshold ({self.detect_thresh})")
            return torch.tensor([]), torch.tensor([])

        print(f"Top Detection Score: {top_score.item()}")

        # Normalize box coords (x1, y1, x2, y2) by width and height
        h, w = target_image.height, target_image.width
        norm_factors = torch.tensor([w, h, w, h], device=top_box.device)
        top_box = top_box / norm_factors

        top_box = top_box.unsqueeze(0)
        top_score = top_score.unsqueeze(0)

        if plot_result:
            # target_img_unnorm = get_preprocessed_image(target_img_processed)
            self.plot_image_with_bbox(target_image, top_box.detach().cpu().numpy(), top_score.detach().cpu().numpy())

        return top_score, top_box



    @torch.inference_mode()
    def predict(self, image: np.ndarray):

        scores, boxes = self.is_query_in_image(image)

        detection = ObjectDetections(
            boxes = boxes.detach().cpu(),                                # torch
            logits = scores.detach().cpu(),                              # torch scalar
            phrases = self.query_txt,
            image_source = image,
            fmt = "xyxy"
        )
        return detection


    # ----- Utils ------


    @staticmethod
    def _ensure_pil(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:

        #If PIL
        if isinstance(img, Image.Image):
            return img.convert("RGB")

        #If Numpy Array
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            return Image.fromarray(img.astype(np.uint8)).convert("RGB")

        #If Torch Tensor
        if isinstance(img, torch.Tensor):
            # Accept CHW [0..1] or [0..255], or HWC
            t = img.detach().cpu()
            if t.ndim == 3:
                if t.shape[0] in (1,3):  # CHW
                    t = t.mul(255.0) if t.max() <= 1.0 else t
                    t = t.byte().clamp(0,255)
                    t = t.permute(1,2,0).numpy()
                else:  # HWC
                    t = t.mul(255.0) if t.max() <= 1.0 else t
                    t = t.byte().clamp(0,255).numpy()
            else:
                raise ValueError("Expected 3D tensor image (CHW or HWC).")
            return Image.fromarray(t).convert("RGB")
        raise TypeError("Unsupported image type. Use PIL.Image, numpy array, or torch.Tensor.")

    def plot_image_with_bbox(self, image_pil, boxes, similarities):

        assert len(boxes) == len(similarities), "Please provide similarity values corresponding to the boxes. Should be equal length lists."

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_axis_off()

        # ax.imshow(image_pil, extent=(0, 1, 1, 0))
        ax.imshow(image_pil)

        # Normalize (x1, y1, x2, y2) by width and height
        h, w = image_pil.height, image_pil.width
        norm_factors = np.array([w, h, w, h])

        for sim, box_xyxy in zip(similarities, boxes):


            box_xyxy = box_xyxy * norm_factors

            x1, y1, x2, y2 = [float(v) for v in box_xyxy]

            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                            fill=False, linewidth=2))

            # add similarity score text above the box
            ax.text(
                x1, y1 - 5, f"{sim:.3f}",
                fontsize=10, color="white", backgroundcolor="red"
            )


        ax.set_title(f'Top {len(boxes)} objects by Similarity')
        plt.show()
    


### Hosting the detector on a local server
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image
from typing import Optional


class OwlViT_Client:
    def __init__(self, port: int = 12181):
        self.url = f"http://localhost:{port}/owlvit"

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = "") -> ObjectDetections:
        response = send_request(self.url, image=image_numpy, caption=caption)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    args = parser.parse_args()

    print("Loading OwLViT Detector model...")

    class OwLViT_Server(ServerMixin, OwlViT_Detector):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image).to_json()

    owlvit = OwLViT_Server()

    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(owlvit, name="owlvit", port=args.port)