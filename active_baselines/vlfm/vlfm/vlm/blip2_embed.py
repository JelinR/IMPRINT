from typing import Any, Optional, List
import torch
import numpy as np

from PIL import Image
from lavis.models import load_model_and_preprocess

#Model for extract corresponding embedding given an image or a text (or both)
class BLIP2_Embed:
    def __init__(
            self,
            model_name: str = "blip2_feature_extractor",
            model_type: str = "pretrain",
            device: Optional[Any] = None,
    ):
        
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model, self.vis_processors, self.txt_processors = \
            load_model_and_preprocess(name=model_name, 
                                        model_type=model_type, 
                                        is_eval=True, 
                                        device=device)

    
    def get_embed(self, image: np.ndarray = None, txt: str = None, debug: bool = False):

        assert image is not None or txt is not None, "Both input image and text is None. Please provide at least one of either."
        
        sample = {}
        if image is not None:
            image = Image.fromarray(image).convert('RGB')
            sample['image'] = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
            mode = 'image'

        if txt is not None:
            sample['text_input'] = [ self.txt_processors["eval"](txt) ]
            if 'image' in sample.keys(): mode = 'multimodal'
            else: mode = 'text'

        if debug: print(f'Mode: {mode}')

        #Extract features, and obtain unimodal or multimodal embedding, and take the mean along the query dimension
        #Shape Example: (1, 32, 768)
        features = self.model.extract_features(sample, mode = mode)
        embed = getattr(features, f'{mode}_embeds')

        # Projecting to lower dimensional space (768 -> 256)
        # embed = getattr(features, f'{mode}_embeds' if mode == 'multimodal' else f'{mode}_embeds_proj')

        #Return with shape: (1, 768)
        return embed.mean(dim=1)


    # def get_embed_eval(self, imgs: List[np.ndarray] = None, txt: str = None):

    #     assert imgs is not None or txt is not None, "Both input image and text is None. Please provide at least one of either."

    #     if imgs is not None:

    #         embed = None
    #         for img in imgs:
    #             if embed is None:
    #                 embed = self.get_embed(image = img, txt = None)
                
    #             else:
    #                 embed += self.get_embed(image = img, txt = None)

    #         mean_embed = embed / len(imgs)
    #         return mean_embed

    #     return self.get_embed(txt = txt)
