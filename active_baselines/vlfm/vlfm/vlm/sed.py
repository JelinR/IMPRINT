import argparse
import os
import sys
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

from sed.sed_model import SED
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image
from detectron2.structures import ImageList
from sed import add_sed_config
import open_clip
import torch.nn.functional as F

unravel = lambda ind, shape: np.unravel_index(ind, shape)

class SED_Model:

    def __init__(self):
        self.args = get_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        og_path = os.path.abspath('.')
        repo_path = os.path.abspath('./SED')

        os.chdir(repo_path)
        print("Changed dir to: ", repo_path)

        # Step 1: Setup configuration
        self.cfg = setup_cfg(self.args)

        # Step 2: Load SED model
        self.model = load_sed_model(self.cfg, self.args.model_path).to(self.device)

        os.chdir(og_path)
        print("Changed dir to: ", og_path)

    def get_patch_embeds(self, img_rgb):
        with torch.no_grad():
            embeds = self.model.extract_clip_patch_features(img_rgb = img_rgb)

        return embeds

# New SEDFeat class for patch features extraction
class SEDFeat(SED):
    def __init__(self, *args, **kwargs):
        super(SEDFeat, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def extract_clip_patch_features(self, img_rgb):
        # images = img

        # images = load_image(img_path).to(self.device)

        images = torch.as_tensor(np.ascontiguousarray(img_rgb.transpose(2, 0, 1))).unsqueeze(0)
        images = images.to(self.device)


        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        clip_images = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        clip_vis_dense = clip_features["clip_vis_dense"]
        return clip_vis_dense

class OpenClip_Embed:

    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.sed_model = SED_Model()  #TODO: Uncomment
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'convnext_large_d_320', pretrained='laion2b_s29b_b131k_ft_soup'
        )
        self.clip_model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer("convnext_large_d_320")

    # Function to extract the text embedding using OpenCLIP
    def get_embed(self, 
                    image: np.ndarray = None, 
                    txt: str = None,
                    top_n_patch_embeds: int = 1):

        if txt is not None:
            text_tokens = self.tokenizer([txt]).to(self.device)

            with torch.no_grad():
                text_embedding = self.clip_model.encode_text(text_tokens)

        if image is None: return text_embedding

        print("\nTop N Patch Embeds: ", top_n_patch_embeds)

        #Get image patch embeddings, and broadcast text embedding
        image_patch_embeds = self.sed_model.get_patch_embeds(img_rgb = image)   #Shape: (1, 768, 24, 24)
        # image_patch_embeds = torch.rand((1, 768, 24, 24)).to(self.device)   #TODO: Remove
        text_ref_embed = text_embedding.unsqueeze(-1).unsqueeze(-1)             #Shape: (1, 768, 1, 1)

        #Get the patch embedding most similar to text embedding
        patch_sim_scores = F.cosine_similarity(image_patch_embeds, text_ref_embed, dim = 1) #Shape: (1, 24, 24)
        
        ############
        
        flat_arg_max = patch_sim_scores.argmax().item()
        arr_arg_max = np.unravel_index(flat_arg_max, patch_sim_scores.shape)

        image_ref_embedding = image_patch_embeds[:, :, arr_arg_max[1], arr_arg_max[2]]  #Shape: (1, 768)
        image_ref_embedding = image_ref_embedding.unsqueeze(-1).unsqueeze(-1)

        patch_sim_scores = F.cosine_similarity(image_patch_embeds, image_ref_embedding, dim = 1)

        ############
        # print(f"Cosine Score: {patch_sim_scores[0, arr_arg_max[1], arr_arg_max[2]]}")

        top_sims, top_flat_inds = torch.topk(patch_sim_scores.view(-1), top_n_patch_embeds)
        top_arr_inds = list(map(lambda ind: unravel(int(ind), patch_sim_scores.shape), top_flat_inds))

        image_embedding = None
        for ind in top_arr_inds:
            if image_embedding is None:
                image_embedding = image_patch_embeds[:, :, ind[1], ind[2]]
                continue

            image_embedding += image_patch_embeds[:, :, ind[1], ind[2]]

        image_embedding = image_embedding / len(top_arr_inds)
        return image_embedding

        

# Function to set up the Detectron2 configuration
def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sed_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

# Function to load the SED model
def load_sed_model(cfg, model_path):
    model_config = SEDFeat.from_config(cfg)
    model = SEDFeat(**model_config)
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    return model

# Function to load an image and preprocess it
def load_image(image_path):
    image_rgb = read_image(image_path, format="BGR").copy()
    tensor_image = torch.as_tensor(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).unsqueeze(0)
    return tensor_image

# Function to extract the text embedding using OpenCLIP
def extract_text_embedding(text, device):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'convnext_large_d_320', pretrained='laion2b_s29b_b131k_ft_soup'
    )
    tokenizer = open_clip.get_tokenizer('convnext_large_d_320')
    clip_model.to(device)
    text_tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)
    
    return text_embedding

# Argument parser to handle command line arguments
def get_parser():
    parser = argparse.ArgumentParser(description="SED model and CLIP text embedding demo")
    parser.add_argument("--config-file", default="configs/convnextL_768.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--input", default="imgs/lv.jpg", help="Path to input image")
    parser.add_argument("--output", default="outs/", help="Path to save output visualizations")
    parser.add_argument("--text", nargs="+", default=["sofa", "tv", "plant"], help="List of objects for CLIP embedding")
    parser.add_argument("--model-path", default="ckpt/sed_model_large.pth", help="Path to the SED model weights")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Additional config options")

    return parser

def get_args():
    parser = argparse.ArgumentParser(description="SED model and CLIP text embedding demo")
    parser.add_argument("--config-file", default="configs/convnextL_768.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--input", default="imgs/lv.jpg", help="Path to input image")
    parser.add_argument("--output", default="outs/", help="Path to save output visualizations")
    parser.add_argument("--text", nargs="+", default=["sofa", "tv", "plant"], help="List of objects for CLIP embedding")
    parser.add_argument("--model-path", default="ckpt/sed_model_large.pth", help="Path to the SED model weights")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Additional config options")

    args = parser.parse_args([])

    return args


