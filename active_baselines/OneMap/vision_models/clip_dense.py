# numpy
import numpy as np

# modeling
from vision_models.base_model import BaseModel
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
from torch.nn import functional as F
from fvcore.common.checkpoint import Checkpointer
import open_clip
import torch
import cv2

from torchvision import transforms

# Visualization
import matplotlib.pyplot as plt

# typing
from typing import List, Optional

try:
    import tensorrt as trt
except:
    print("TensorRT not available, cannot use Jetson")

from eval.scrape_img import load_images

class ClipModel(torch.nn.Module, BaseModel):
    def __init__(self,
                 path: str,
                 jetson: bool = False,
                 scrape_num: int = 15,
                 ):
        super(ClipModel, self).__init__()
        self.jetson = jetson

        self.input_format = "RGB"
        self.scrape_num = scrape_num

        self.aug = T.ResizeShortestEdge(
            [640, 640], 2560
        )
        self.tokenizer = open_clip.get_tokenizer('convnext_large_d_320')
        self.feature_dim = 768
        self.clip_resolution = (768, 768)

        if self.jetson:
            logger = trt.Logger(trt.Logger.WARNING)
            # Load jetson specific model
            runtime = trt.Runtime(logger)
            with open("trt/clip_model_trt.engine", "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            # self.cfx = cuda.Device(0).retain_primary_contwex()
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.current_stream(torch.device("cuda"))
            with open("trt/clip_text_model_trt.engine", "rb") as f:
                self.engine_txt = runtime.deserialize_cuda_engine(f.read())
            self.context_txt = self.engine_txt.create_execution_context()
            # self.stream_txt = cuda.Stream()

        else:
            name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name,
                pretrained=pretrain,
                device="cuda", )
            checkpointer = Checkpointer(clip_model)
            checkpointer.load(path)
            self.clip_model = clip_model.float()
            self.clip_model.eval()
            self.clip_mean = torch.Tensor([122.7709383, 116.7460125, 104.09373615]).to("cuda")
            self.clip_mean = self.clip_mean.unsqueeze(-1).unsqueeze(-1)
            self.clip_std = torch.Tensor([68.5005327, 66.6321579, 70.3231630]).to("cuda")
            self.clip_std = self.clip_std.unsqueeze(-1).unsqueeze(-1)

    def eval(self):
        super().eval()
        if self.jetson:
            pass
        else:
            self.clip_model.eval()
        return self

    def compute_similarity(self,
                           image_feats: torch.Tensor,
                           text_feats: torch.Tensor,
                           ) -> torch.Tensor:
        # image_feats = F.normalize(image_feats, dim=1)  # B C H W, normalize along C
        # text_feats = F.normalize(text_feats, dim=1)
        # print(image_feats.max())
        # print(text_feats.max())
        if len(image_feats.shape) == 3:
            return torch.einsum('bcx, bc -> bx', image_feats, text_feats)
        else:
            return torch.einsum('bchw, bc -> bhw', image_feats, text_feats) #Shapes: (1, 768, 600, 600), (1, 768) -> (1, 600, 600)


    def get_multi_features(self,                                                #TODO Added: IMPRINT
                                text_feats: torch.Tensor,    # (1, C)
                                text_query: str,
                                scrape_data_dir: str = "/home/akkara/data/data/scraped_imgs/ovon_15",
                                topk: int = 1):
        device = text_feats.device
        B = self.scrape_num

        print(f"Topk : {topk}")
        print(f"Scrape Num: {B}")

        # 1) SCRAPE & PREPROCESS → a float tensor (B, C, H, W) on `device`
        raw_imgs = load_images(query=text_query, num_images=B, save_dir=scrape_data_dir)
        # raw_imgs: list of np.uint8 H×W×3 BGR arrays
        preprocess = transforms.Compose([
            transforms.ToTensor(),             # → [0,1] RGB, shape (3, H, W)
            transforms.Resize(640),            # shorter side → 640
            transforms.CenterCrop(640),        # → (3, 640, 640)
        ])
        imgs_t = torch.stack([ preprocess(img[..., ::-1]) for img in raw_imgs ], dim=0)
        # imgs_t = imgs_t.to(device)             # (B, 3, 640, 640)

        imgs_t = (imgs_t * 255.0).clamp(0,255)
        print(f"Image tensor range : min {imgs_t.min()}, max {imgs_t.max()}")

        # 2) EXTRACT PATCHED IMAGE FEATURES → (B, C, Hp, Wp) e.g. (15, 768, 24, 24)
        scraped_patch_feats = self.get_image_features(imgs_t.cpu().numpy()).to(device = device)
        # print(scraped_patch_feats.shape)

        # 3) COMPUTE COSINE SIMILARITY PER PATCH → (B, Hp, Wp)
        #    text_feats: (C,) or (1,C) → make (1, C, 1, 1) so it broadcasts
        text_ref = text_feats.view(1, -1, 1, 1)  # (1, C, 1, 1)
        patch_sim = F.cosine_similarity(
            scraped_patch_feats,                # (B, C, Hp, Wp)
            text_ref,                           # (1, C, 1, 1) broadcast → (B, C, Hp, Wp)
            dim=1
        )  # → (B, Hp, Wp)
        # print(patch_sim.shape)

        # 4) FLATTEN & TOP‑K → get flat indices of the k best patches per image
        B, Hp, Wp = patch_sim.shape
        flat = patch_sim.view(B, -1)             # (B, Hp*Wp)
        topk_vals, topk_inds = flat.topk(topk, dim=1, largest=True, sorted=False)  # both (B, topk)
        # print(topk_vals.shape, topk_inds.shape)

        # 5) CONVERT FLAT → (row, col)
        row_idx = topk_inds // Wp                # (B, topk)
        col_idx = topk_inds %  Wp                # (B, topk)

        # 6) GATHER THE TOP‑K PATCH FEATURES → (B, C, topk)
        #    Using advanced indexing: feats[B_i, :, row_idx[i], col_idx[i]]
        batch_idx = torch.arange(B, device=device)[:, None]     # (B, 1)
        # row_idx, col_idx are (B, topk)
        selected = scraped_patch_feats[                          # → (B, topk, C)
            batch_idx,                                          # B‑dim
            :,                                                  # C‑dim
            row_idx,                                            # Hp‑dim
            col_idx                                             # Wp‑dim
        ]
        # print(selected.shape)

        # 7) POOL & CONCAT TEXT VECTOR → (B+1, C)
        scraped_img_feats = selected.mean(dim=1)                    # (B, C)   
        scraped_img_feats = F.normalize(scraped_img_feats, dim=1)   # (B, C)

        # scraped_img_feats = scraped_img_feats.mean(dim=0, keepdim=True)     #(1, C). First take the mean of the image features
        # scraped_img_feats = F.normalize(scraped_img_feats, dim=-1)          #(1, C)
        # print(f"Alt Mean scraped_img_feats")
                    
        all_feats = torch.cat([scraped_img_feats, text_feats], dim=0)  # (B+1, C)
        all_feats = all_feats.mean(dim=0, keepdim=True)                # (1, C)
        all_feats = F.normalize(all_feats, dim=1)                      # (1, C)

        return all_feats      #Mean over all image+text embeds

        

    # def forward(self, images: np.ndarray):
    #    return self.image_forward_torch(images)

    # def forward(self, text_tokenized: torch.Tensor):
    # print(text_tokenized.shape)
    # with torch.no_grad():
    # class_embeddings = self.clip_model.encode_text(text_tokenized)
    # return F.normalize(class_embeddings, dim=1)

    def forward_im(self, images: torch.Tensor):
        return self.image_forward_torch(images)

    def forward_text(self, text_tokenized):
        with torch.no_grad():
            class_embeddings = self.clip_model.encode_text(text_tokenized)
            return F.normalize(class_embeddings, dim=1)

    # def forward_text_trt(self, text_tokenized):
    #
    #
    #
    # #class_embeddings = self.clip_model.encode_text(text_tokenized)
    #
    #
    #
    #
    # return F.normalize(torch.tensor(output), dim=1)

    def image_forward_torch(self, clip_images: torch.Tensor):
        with torch.no_grad():
            clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear',
                                        align_corners=False, )
            clip_images = (clip_images - self.clip_mean) / self.clip_std
            clip_features = self.clip_model.encode_image(clip_images, dense=True)
            clip_vis_dense = clip_features["clip_vis_dense"]

            return F.normalize(clip_vis_dense, dim=1)

    def text_forward_trt(self, texts: torch.Tensor):
        # print(texts)
        output_shape = self.engine_txt.get_binding_shape(1)
        # output = np.empty(self.engine_txt.get_binding_shape(1), dtype=np.float32)
        input_tensor = texts.cuda()
        output_tensor = torch.empty(*output_shape, dtype=torch.float32, device="cuda")
        d_input = input_tensor.data_ptr()
        d_output = output_tensor.data_ptr()

        self.context_txt.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=self.stream.cuda_stream)
        # cuda.memcpy_dtoh_async(output, d_output, self.stream_txt)
        # self.stream_txt.synchronize()
        # d_output.free()
        # d_input.free()
        # print(output[:, 0])
        # print(output.sum())
        return F.normalize(output_tensor, dim=1).to(torch.float32)

    def image_forward_trt(self, input_tensor: torch.Tensor):
        # print(f"Input shape: {images.shape}, dtype: {images.dtype}")
        # print(f"Input binding shape: {self.engine.get_binding_shape(0)}")
        # print(f"Output binding shape: {self.engine.get_binding_shape(1)}")
        # TODO: likely the amount of needed copies can be reduced here
        # input_tensor = torch.tensor(images, dtype=torch.float32, device="cuda")

        output_shape = self.engine.get_binding_shape(1)
        output_tensor = torch.empty(*output_shape, dtype=torch.float32, device="cuda")
        d_output = output_tensor.data_ptr()
        d_input = input_tensor.data_ptr()
        # cuda.memcpy_htod_async(d_input, images , torch.cuda.current_stream())
        bindings = [int(d_input)] + [int(d_output)]
        # self.cfx.push()
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)
        # self.cfx.pop()
        # cuda.memcpy_dtoh_async(output, d_output, torch.cuda.current_stream().current_stream)
        # self.stream.synchronize()
        # d_output.free()
        # d_input.free()
        return F.normalize(output_tensor, dim=1).to(torch.float32)

    def get_image_features(self,
                           images: np.ndarray
                           ) -> torch.Tensor:
        # expects images in shape B C H W in BGR, expected to be a numpy array
        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "BGR":
                # whether the model expects BGR inputs or RGB
                original_image = images[:, ::-1, :, :]
            else:
                original_image = images
            to_transform_img = original_image.transpose(0, 2, 3, 1)
            transformed_0 = self.aug.get_transform(to_transform_img[0]).apply_image(to_transform_img[0]).transpose(2, 0,
                                                                                                                   1)
            transformed = np.zeros((to_transform_img.shape[0], *transformed_0.shape), dtype=transformed_0.dtype)
            transformed[0] = transformed_0
            # TODO Can we do this batchwise?
            for i in range(1, to_transform_img.shape[0]):
                transformed[i] = self.aug.get_transform(to_transform_img[i]).apply_image(to_transform_img[i]).transpose(
                    2, 0, 1)
            # print(transformed.shape)

            # After, differentiate between jetson and normal
            if self.jetson:
                # images = np.ascontiguousarray(transformed).astype(np.float32)
                # print(images.dtype)
                transformed = F.interpolate(torch.as_tensor(transformed.astype("float32")).to("cuda"),
                                            size=self.clip_resolution, mode='bilinear',
                                            align_corners=False, )
                return self.image_forward_trt(transformed)
            else:
                # images = torch.as_tensor(transformed.astype("float32")).to("cuda")
                transformed = torch.as_tensor(transformed.astype("float32")).to("cuda")

                return self.image_forward_torch(transformed)

    def get_text_features(self,
                          texts: List[str]
                          ) -> torch.Tensor:
        with torch.no_grad():
            texts = self.tokenizer(texts)

            # print(texts.shape)
            # print(texts.dtype)
            # After, differentiate between jetson and normal

            if self.jetson:
                return self.text_forward_trt(texts)
            else:
                class_embeddings = self.clip_model.encode_text(texts.to("cuda"))
                return F.normalize(class_embeddings, dim=1)

def resize_and_pad(img, target_size, pad_value=0):
    """
    Scales img to fit within target_size, then pads the shorter dimension.
    """
    H, W = target_size
    h, w = img.shape[:2]
    scale = min(H/h, W/w)
    new_h, new_w = int(h*scale), int(w*scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # compute padding
    pad_vert = H - new_h
    pad_horiz = W - new_w
    top = pad_vert // 2
    bottom = pad_vert - top
    left = pad_horiz // 2
    right = pad_horiz - left

    padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value
    )
    return padded

def batch_resize_and_pad(img_list, target_size, pad_value=0):
    """
    Applies resize_and_pad to each image in img_list and
    stacks them into a single NumPy array of shape (B, H, W, C).
    
    Args:
        img_list (List[np.ndarray]): List of H×W×C images.
        target_size (tuple): Desired (H_target, W_target).
        pad_value (int or tuple): Pixel value for padding.

    Returns:
        np.ndarray: Array of shape (B, H_target, W_target, C).
    """
    padded_imgs = []
    for img in img_list:
        padded = resize_and_pad(img, target_size, pad_value)
        # If grayscale, add channel dimension
        if padded.ndim == 2:
            padded = padded[:, :, None]
        padded_imgs.append(padded)

    batch = np.stack(padded_imgs, axis=0)
    return batch

def topk_patch_indices(scores: torch.Tensor, k: int = 20):
    """
    Args:
        scores: Tensor of shape (B, H, W), patch scores per image.
        k:      number of top patches to select for each image.

    Returns:
        row_idx, col_idx: two LongTensors of shape (B, k), where
            row_idx[i, j], col_idx[i, j] are the coordinates of the j-th
            highest-scoring patch in image i.
    """
    B, H, W = scores.shape
    # flatten to (B, H*W)
    flat = scores.view(B, -1)               # (B, H*W)

    # topk returns sorted values & indices along dim=1
    topk_vals, topk_inds = flat.topk(k, dim=1, largest=True, sorted=True)  # both (B, k)

    # unravel into 2D indices
    row_idx = topk_inds // W                # integer division
    col_idx = topk_inds %  W                # remainder

    return row_idx, col_idx


if __name__ == "__main__":
    import time
    use_jetson = False
    N = 1
    import cv2
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    clip = ClipModel('../weights/clip.pth', use_jetson) # Jetson
    img = read_image('rgb.jpg', format="RGB")
    # img = read_image('/home/finn/drafting/CLIPTest/sim2.png', format="RGB")
    # img = read_image('/home/Pictures/chair.png', format="RGB")
    # img = read_image('/home/spot/chair.png', format="RGB")
    # img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)
    img_feats_ = clip.get_image_features(img)
    # print(img_feats_)
    # text_feats = clip.encode_text("A photo of a robot endeffector")

    # start.record()
    # Perform N iterations and measure overall time
    print("a")
    start_time = time.time()
    for i in range(N):
        # img[:, i*5:(i+1*5)] -= i
        img_feats = clip.get_image_features(img)
        # print(img_feats.sum())
        # print(img_feats.sum())
        torch.cuda.synchronize()  # Synchronize after each forward pass

    end_time = time.time()
    # Compute overall time and average time per iteration
    total_time = end_time - start_time
    avg_time_per_iteration = total_time / N
    # end.record()

    print(f"Total time for {N} iterations: {total_time:.4f} seconds")
    print(f"Average time per iteration: {avg_time_per_iteration:.4f} seconds")    # clip = ClipModel('weights/clip.pth', False) # Jetson
    txt_feats = clip.get_text_features(["a chair"])
    sim = clip.compute_similarity(img_feats, txt_feats)
    print(sim.max(), sim.min(), sim.mean())
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(sim[0].detach().cpu())
    axs[1].imshow(img.transpose(1, 2, 0))
    plt.savefig("plant.png")
    plt.show()
    # print(img_feats.shape, text_feats.shape)
