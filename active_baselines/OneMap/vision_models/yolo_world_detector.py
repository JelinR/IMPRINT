# typing
from typing import List, Dict

import numpy as np
# inference
from inference.models import YOLOWorld
#from inference.models import YOLOWorld, YOLOWorld_Multi
from inference.core import logger
from inference.core.utils.hash import get_text_hash
from inference.core.cache import cache
import torch
from eval.scrape_img import load_images

EMBEDDINGS_EXPIRE_TIMEOUT = 1800  # 30 min

# cv2
import cv2

# supervision
import supervision as sv


class YOLOWorldDetector:
    def __init__(self,
                 confidence_threshold: float,
                 multi_prompt: bool = False
                 ):

        if multi_prompt:
            self.model = YOLOWorld_Multi(model_id="yolo_world/l")
            print("Detector : Multi Mode")
        else:
            self.model = YOLOWorld(model_id="yolo_world/l")

        print(f"\nSetting YOLOWorld Conf Thresh : {confidence_threshold}\n")
        self.confidence_threshold = confidence_threshold
        self.classes = None

    def set_classes(self,
                    classes: List[str]
                    ):
        self.classes = classes
        self.model.set_classes(classes)

        # num_classes = len(classes)

        # self.model.nc    = num_classes
        # self.model.names = {i: name for i, name in enumerate(classes)}
        # # mirror into the internal Ultralytics model as well
        # self.model.model.nc    = num_classes
        # self.model.model.names = self.model.names

    def detect(self, image: np.ndarray) -> dict:
        if self.classes is None:
            raise ValueError("Classes must be set before detecting")

        results = self.model.infer(image, confidence=self.confidence_threshold)

        preds = {
            "boxes": [],
            "scores": []
        }

        for detection in results.predictions:
            cls = detection.class_id
            class_name = detection.class_name

            if class_name == self.classes[0] and detection.confidence > self.confidence_threshold:
                x1 = detection.x - detection.width / 2
                y1 = detection.y - detection.height / 2
                x2 = detection.x + detection.width / 2
                y2 = detection.y + detection.height / 2

                # Check if box is not a point
                if x1 != x2 and y1 != y2:
                    preds["boxes"].append([x1, y1, x2, y2])
                    preds["scores"].append(detection.confidence)

        return preds

class YOLOWorld_Multi(YOLOWorld):
    
    def __init__(self, *args, model_id="yolo_world/l", **kwargs):
        super().__init__(*args, model_id=model_id, **kwargs)

        self.scrape_num = 3
    
    def load_image_embeds(self, text: str,
                            scrape_data_dir = "/home/akkara/data/data/scraped_imgs/ovon_15"):

        scraped_imgs = load_images(query = text,
                                num_images = self.scrape_num,
                                save_dir= scrape_data_dir)

        image_embeds = self.clip_model.embed_image(scraped_imgs)    #Shape: (3, 512)
        return image_embeds


    def set_classes(self, text: list):
        """Set the class names for the model.

        Args:
            text (list): The class names.
        """
        class_names_to_calculate_embeddings = []
        classes_embeddings = {}
        for class_name in text:
            class_name_hash = f"clip-embedding:{get_text_hash(text=class_name)}"
            embedding_for_class = cache.get_numpy(class_name_hash)
            if embedding_for_class is not None:
                logger.debug(f"Cache hit for class: {class_name}")
                classes_embeddings[class_name] = embedding_for_class
            else:
                logger.debug(f"Cache miss for class: {class_name}")
                class_names_to_calculate_embeddings.append(class_name)
        if len(class_names_to_calculate_embeddings) > 0:
            logger.debug(
                f"Calculating CLIP embeddings for {len(class_names_to_calculate_embeddings)} class names"
            )

            #TODO CHANGED: Commented this out and replaced with average of multiple embeddings
            # cache_miss_embeddings = self.clip_model.embed_text(
            #     text=class_names_to_calculate_embeddings
            # )

            text_embed = self.clip_model.embed_text(text=class_names_to_calculate_embeddings)
            avg_embed = None

            for count, name in enumerate(class_names_to_calculate_embeddings):
                image_embeds = self.load_image_embeds(name)                                                 #Shape: (3, 512)
                multi_embeds = np.concatenate((text_embed[count][np.newaxis, ...], image_embeds), axis=0)   #Shape: (4, 512)
                avg_embed_curr = np.mean(multi_embeds, axis=0, keepdims=True)                               #Shape: (1, 512)

                if avg_embed is None:
                    avg_embed = avg_embed_curr
                else:
                    avg_embed = np.concatenate((avg_embed, avg_embed_curr), axis=0)


            cache_miss_embeddings = avg_embed


        else:
            cache_miss_embeddings = []
        for missing_class_name, calculated_embedding in zip(
            class_names_to_calculate_embeddings, cache_miss_embeddings
        ):
            classes_embeddings[missing_class_name] = calculated_embedding
            missing_class_name_hash = (
                f"clip-embedding:{get_text_hash(text=missing_class_name)}"
            )
            cache.set_numpy(  # caching vectors of shape (512,)
                missing_class_name_hash,
                calculated_embedding,
                expire=EMBEDDINGS_EXPIRE_TIMEOUT,
            )
        embeddings_in_order = np.stack(
            [classes_embeddings[class_name] for class_name in text], axis=0
        )
        txt_feats = torch.from_numpy(embeddings_in_order)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.model.model.txt_feats = txt_feats.reshape(
            -1, len(text), txt_feats.shape[-1]
        ).detach()
        self.model.model.model[-1].nc = len(text)

        ###TODO ADDED: Fixed the bug for negative dimension -73
        # 1) Update the internal Model instance
        num_classes = len(text)

        self.model.model.nc    = num_classes
        self.model.model.names = {i: name for i, name in enumerate(text)}

        ####

        self.class_names = text



if __name__ == "__main__":
    # Test the YOLO World Detector
    detector = YOLOWorldDetector(confidence_threshold=0.5)
    detector.set_classes(["person", "car", "truck", "bus", "bicycle", "motorbike", "traffic light", "stop sign"])

    # Load an image
    image = cv2.imread("test_images/a.jpg")

    # Detect objects in the image
    detections = detector.detect(image)

    # Display the image with the detections
    image_with_detections = detections.draw_on_image(image)
    cv2.imshow("Detections", image_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
