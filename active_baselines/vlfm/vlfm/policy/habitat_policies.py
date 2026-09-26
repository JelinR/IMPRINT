# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch
import gzip
import json
import pandas as pd
from depth_camera_filtering import filter_depth
from frontier_exploration.base_explorer import BaseExplorer
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default_structured_configs import (
    PolicyConfig,
)
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat.config import read_write
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch import Tensor

# Required for GUI Nav
from habitat.utils.visualizations import maps
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import os
import heapq
from typing import Tuple
from vlfm.vlm.detections import ObjectDetections as vlm_obj_detection

from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix, closest_point_within_threshold, transform_points
from vlfm.vlm.grounding_dino import ObjectDetections
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer

from ..mapping.obstacle_map import ObstacleMap
from .base_objectnav_policy import BaseObjectNavPolicy, VLFMConfig
from .itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3, ITM_Scrape_Policy, ITM_Scrape_Grid_Policy

from eval.sim_grid import Sim_Grid_Online
from eval.utils.scrape_img import load_images

try:
    from vlfm.mapping.sed_embed_map import Obstacle_Embed_SED_Map
except ModuleNotFoundError:
    print("Could not load SED Obstacle Map Module. This is fine if using BLIP2 for embeddings.")

try:
    from vlfm.mapping.blip_embed_map import Obstacle_Embed_BLIP_Map
except ModuleNotFoundError:
    print("Could not load BLIP2 Obstacle Map Module. This is fine if using SED for embeddings.")

try:
    from ovon.ovon.task.sensors import ClipObjectGoalSensor
    from ovon.utils.utils import load_pickle
except ModuleNotFoundError:
    print(f"Could not load OVON Module. This is fine if you are not using OVON ObjectNav Dataset.")


try:
    from vlfm.vlm.owlvit import OwlViT_Detector, Owlv2_Detector_t, Owlv2_Detector_img_cond
    from vlfm.vlm.coco_classes import COCO_CLASSES
except:
    print(f"Could not import OwlViT Detector.")

HM3D_ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]
MP3D_ID_TO_NAME = [
    "chair",
    "table|dining table|coffee table|side table|desk",  # "table",
    "framed photograph",  # "picture",
    "cabinet",
    "pillow",  # "cushion",
    "couch",  # "sofa",
    "bed",
    "nightstand",  # "chest of drawers",
    "potted plant",  # "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv",  # "tv monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]

#Time Loading Sim Dict
import time
save_times_path = "/mnt/vlfm_query_embed/timings/nav/load_sim_grid/text/blip/times_grid_1.txt"
os.makedirs(os.path.dirname(save_times_path), exist_ok=True)


import hashlib

def array_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()

class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class HabitatMixin:
    """This Python mixin only contains code relevant for running a BaseObjectNavPolicy
    explicitly within Habitat (vs. the real world, etc.) and will endow any parent class
    (that is a subclass of BaseObjectNavPolicy) with the necessary methods to run in
    Habitat.
    """

    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # must be set by _reset() method
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = False

    OVON_ID_TO_NAME: Dict = {}
    HSSD_ID_TO_CAT: Dict = {}

    def __init__(
        self,
        camera_height: float,
        min_depth: float,
        max_depth: float,
        camera_fov: float,
        image_width: int,
        dataset_type: str = "hm3d",
        gui_nav: bool = False,
        saved_nav: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        camera_fov_rad = np.deg2rad(camera_fov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
        self._dataset_type = dataset_type

        self.gui_nav = gui_nav
        self.saved_nav = saved_nav
        self.step_actions = None

        if self.gui_nav and self.saved_nav:
            print(f"WARNING: Both gui_nav and saved_nav are set to True. Ignoring saved_nav argument...")

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "HabitatMixin":
        policy_config: VLFMPolicyConfig = config.habitat_baselines.rl.policy
        
        #Loads all the relevant configs (inside VLFMPolicyConfig)
        #VLFMPolicyConfig is defined in base_objectnav_policy.py
        kwargs = {k: policy_config[k] for k in VLFMPolicyConfig.kwaarg_names}  # type: ignore

        # In habitat, we need the height of the camera to generate the camera transform
        sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]

        # Synchronize the mapping min/max depth values with the habitat config
        kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
        kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
        kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
        kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width

        # Only bother visualizing if we're actually going to save the video
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0

        if "ovon" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "ovon"
            ovon_cat_to_embed_path = f"habitat-lab/{config.habitat.task.lab_sensors.clip_objectgoal_sensor.cache}"
            ovon_cat_to_embed = load_pickle(ovon_cat_to_embed_path)
            cls.OVON_ID_TO_NAME = {array_hash(v): k for k, v in ovon_cat_to_embed.items()}
        elif "hm3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "hm3d"
        elif "mp3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "mp3d"
        elif "hssd" in config.habitat.dataset.data_path:        #Added for HSSD
            kwargs["dataset_type"] = "hssd"

            # with gzip.open(f"habitat-lab/{config.habitat.dataset.data_path}", "rt") as f:
            #     hssd_data_info = json.load(f)

            # hssd_cat_to_id = hssd_data_info["category_to_task_category_id"]
            # cls.HSSD_ID_TO_CAT = {id: cat for cat, id in hssd_cat_to_id.items()}

            hssd_root_dir = os.path.dirname(config.habitat.dataset.scenes_dir)
            obj_rare_path = os.path.join("habitat-lab", hssd_root_dir, "semantics/rare_objects.csv")
            obj_rare_df = pd.read_csv(obj_rare_path, index_col = 0)

            cls.HSSD_ID_TO_CAT = obj_rare_df["rare_category"].to_dict()
        else:
            raise ValueError("Dataset type could not be inferred from habitat config")

        if config.habitat_baselines.rl.policy.name.__contains__("HabitatGridPolicy"):
            kwargs["nav_with_grid"] = True
        else:
            kwargs["nav_with_grid"] = False
        
        assert kwargs["embed_model_name"] in ["BLIP2", "SED", "clip", "siglip"], "Please provide a valid embed model name: ['BLIP2', 'SED']"

        # if config.grid_embed.nav_with_grid:
        #     kwargs["nav_with_grid"] = True
        # else:
        #     kwargs["nav_with_grid"] = False

        if config.grid_embed.gui_nav:
            kwargs["compute_frontiers"] = False
            kwargs["gui_nav"] = True
        
        elif config.grid_embed.saved_nav:
            kwargs["saved_nav"] = True

        
        if kwargs["prompt_mode"] == "text":
            kwargs["scrape_imgs"] = False
            kwargs["process_sim_mode"] = "mean"
        else:
            kwargs["scrape_imgs"] = True


        return cls(**kwargs)


    def act(
        self: Union["HabitatMixin", BaseObjectNavPolicy],
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
        gui_info = None                 #TODO Added for embed
    ) -> PolicyActionData:

        """Converts object ID to string name, returns action as PolicyActionData"""
        #object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
        obs_dict = observations.to_tree()

        if self._dataset_type == "ovon":
            object_embed = observations[ClipObjectGoalSensor.cls_uuid].cpu().numpy()
            obs_dict["objectgoal"] = self.OVON_ID_TO_NAME[array_hash(object_embed)]
        
        elif self._dataset_type == "hm3d":
            object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
            obs_dict[ObjectGoalSensor.cls_uuid] = HM3D_ID_TO_NAME[object_id]
        elif self._dataset_type == "mp3d":
            object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
            obs_dict[ObjectGoalSensor.cls_uuid] = MP3D_ID_TO_NAME[object_id]
            self._non_coco_caption = " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."
        elif self._dataset_type == "hssd":                   
            object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()                                   #Added for HSSD
            # obs_dict[ObjectGoalSensor.cls_uuid] = HM3D_ID_TO_NAME[object_id]
            obs_dict[ObjectGoalSensor.cls_uuid] = self.HSSD_ID_TO_CAT[object_id]
        else:
            raise ValueError(f"Dataset type {self._dataset_type} not recognized")

        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        try:
            action, rnn_hidden_states = parent_cls.act(obs_dict, rnn_hidden_states, prev_actions, masks, deterministic)
        except StopIteration:
            action = self._stop_action
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""

        self._done_initializing = not self._num_steps < 11  # type: ignore
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        parent_cls._reset()
        self._start_yaw = None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """Get policy info for logging"""
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(detections)

        if not self._visualize:  # type: ignore
            return info

        if self._start_yaw is None:
            self._start_yaw = self._observations_cache["habitat_start_yaw"]
        info["start_yaw"] = self._start_yaw
        return info

    def _cache_observations(self: Union["HabitatMixin", BaseObjectNavPolicy], observations: TensorDict) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (TensorDict): The observations from the current timestep.
        """
        if len(self._observations_cache) > 0:
            return
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()

        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        # Habitat GPS makes west negative, so flip y
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        self._obstacle_map: ObstacleMap
        if self._compute_frontiers:
            self._obstacle_map.update_map(   
                depth = depth,
                tf_camera_to_episodic=tf_camera_to_episodic,
                min_depth=self._min_depth,
                max_depth=self._max_depth,
                fx=self._fx,
                fy=self._fy,
                topdown_fov=self._camera_fov
            )
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        else:
            if "frontier_sensor" in observations:
                frontiers = observations["frontier_sensor"][0].cpu().numpy()
            else:
                frontiers = np.array([])

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": observations["depth"],  # for pointnav
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"][0].item(),
        }


@baseline_registry.register_policy
class OracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    def _explore(self, observations: TensorDict) -> Tensor:
        explorer_key = [k for k in observations.keys() if k.endswith("_explorer")][0]
        pointnav_action = observations[explorer_key]
        return pointnav_action


@baseline_registry.register_policy
class SuperOracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,  # can be anything because it is not used
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        return PolicyActionData(
            actions=observations[BaseExplorer.cls_uuid],
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )


@baseline_registry.register_policy
class HabitatITMPolicy(HabitatMixin, ITMPolicy):
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV2(HabitatMixin, ITMPolicyV2):
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV3(HabitatMixin, ITMPolicyV3):
    pass


@baseline_registry.register_policy
class HabitatITMPolicy_owlv2(HabitatMixin, ITMPolicyV2):
    "Standard VLFM : Replacing GroundingDino with OwLViT Detector"
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._object_detector = Owlv2_Detector_t(
                                    detect_thresh = self._non_coco_threshold,
                                    # model_id = "google/owlv2-base-patch16",   
                                    model_id = "google/owlv2-base-patch16-ensemble"
                                    )

    def _reset(self):
        super()._reset()
        self._object_detector.query_txt = None

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:

        super()._pre_step(observations, masks)

        if (self._object_detector.query_txt is None):

            #Query Vector is the text embedding vector corresponding to the target
            print(f"Setting Reference Text Query for Owlv2 Detector...")
            self._object_detector.set_query(texts = [self._target_object])                                 

            print(f"------Query for OWL-ViT Detector is initialized!-----\n\n")
        
    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_object.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img)
        )

        if has_coco: 
            detections.filter_by_class(target_classes)

        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

@baseline_registry.register_policy
class HabitatITMPolicy_owlv2_Scraped(HabitatMixin, ITM_Scrape_Policy):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._object_detector = Owlv2_Detector_t(
                                    detect_thresh = self._non_coco_threshold,
                                    # model_id = "google/owlv2-base-patch16",   
                                    model_id = "google/owlv2-base-patch16-ensemble"
                                    )

    def _reset(self):
        super()._reset()
        self._object_detector.query_txt = None

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:

        super()._pre_step(observations, masks)

        if (self._object_detector.query_txt is None):

            #Query Vector is the text embedding vector corresponding to the target
            print(f"Setting Reference Text Query for Owlv2 Detector...")
            self._object_detector.set_query(texts = [self._target_object])                                 

            print(f"------Query for OWL-ViT Detector is initialized!-----\n\n")
    
    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_object.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img)
        )

        if has_coco: 
            detections.filter_by_class(target_classes)

        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

@baseline_registry.register_policy
class HabitatITMPolicy_owlv2_Scraped_Detect(HabitatMixin, ITM_Scrape_Policy):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._object_detector = Owlv2_Detector_img_cond(model_id = "google/owlv2-base-patch16-ensemble")
        self.detect_scrape_num = 10
        print(f"Detect Thresh : {self._non_coco_threshold}")
        print(f"Detect Scrape num: {self.detect_scrape_num}")
        print(f"Scrape num : {self.scrape_num}")
        
    def _reset(self):
        super()._reset()
        self._object_detector.img_emb = None
        self._object_detector.txt_emb = None
        self._object_detector.query_txt = None

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:

        super()._pre_step(observations, masks)

        if (self._object_detector.query_txt is None):

            #Query Vector is the text embedding vector corresponding to the target
            print(f"Setting Reference Multi-Query for Owlv2 Detector...")
            query_images = load_images(query = self._target_object, 
                                       num_images = self.detect_scrape_num, 
                                       save_dir = self.scrape_data_dir)

            query_vecs = []
            for query_img in query_images:

                self._object_detector.extract_query_vector(texts = [self._target_object],
                                                            source_image = query_img)
                query_vecs.append(self._object_detector.img_emb)

            query_vec = torch.stack(query_vecs, dim=0).mean(dim=0)
            self._object_detector.img_emb = query_vec                                  

            print(f"------Query for OWL-ViT Detector is initialized!-----\n\n")
    
    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_object.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img)
        )

        if has_coco: 
            detections.filter_by_class(target_classes)

        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections


@dataclass
class VLFMPolicyConfig(VLFMConfig, PolicyConfig):
    pass

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

@baseline_registry.register_policy
class HabitatGridPolicy(HabitatMixin, BaseObjectNavPolicy):

    _count_same_pos: int = 0
    _prev_agent_xy: np.ndarray = np.zeros((2))
    _same_pos_thresh: int = 5
    _top_k_preds: int = 3
    nav_to_best_grid: bool = False
    xy_before_nav: Tuple[Any] = None

    def __init__(self, 

        #Sim Grid Args
        prompt_mode: str = "text", 
        process_sim_mode: str = "mean",
        scrape_imgs: bool = False, 
        scrape_num: int = 3, 
        scrape_data_dir: str = "data/scraped_imgs/ovon_15",
        top_n_patch_embeds: int = 1,
        sim_goal_thresh: float = 0.75,

        *args, **kwargs):
            
            super().__init__(*args, **kwargs)
            self.gui_action_t = None
            self.sim_goal_thresh = sim_goal_thresh

            self.top_xy = []
            self.top_sims = []
            self._last_frontier, self._last_value = np.zeros(2), float("-inf")
            self._acyclic_enforcer = AcyclicEnforcer()

            if self._compute_frontiers:
                self.sim_grid = Sim_Grid_Online(model_name = self._obstacle_map.embed_model_name,
                                         use_model = self._obstacle_map._embed_model,
                                         grid_size = self._obstacle_map.cell_size,
                                         prompt_mode = prompt_mode,
                                         process_sim_mode = process_sim_mode,
                                         scrape_imgs = scrape_imgs,
                                         scrape_num = scrape_num,
                                         scrape_data_dir = scrape_data_dir,
                                         top_n_patch_embeds = top_n_patch_embeds)

    def _reset(self) -> None:

        super()._reset()

        self._count_same_pos = 0
        self._prev_agent_xy= np.zeros((2))
        self.nav_to_best_grid= False
        self.xy_before_nav = None

        self.top_xy = None
        self.top_sims = None
        self._last_frontier, self._last_value = np.zeros(2), float("-inf")
        self._acyclic_enforcer = AcyclicEnforcer()


    def act(self,
        observations: TensorDict,
        rnn_hidden_states: Any,  # can be anything because it is not used
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
        gui_info: TensorDict = None,
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        
        #Navigation using GUI
        if self.gui_nav:
            if not self._did_reset and masks[0] == 0: self._reset()

            scene_id = gui_info["scene_id"]
            episode_id = gui_info["episode_id"]

            print(f"\nStep: {self._num_steps}")
            self._num_steps += 1

            if "top_down" not in gui_info:
                plot_frame = observations["rgb"][0].cpu().numpy().astype(np.uint8)[:, :, ::-1]                
            else:
                rgb = observations["rgb"][0].cpu().numpy().astype(np.uint8)[:, :, ::-1] 
                top_down = maps.colorize_draw_agent_and_fit_to_height(gui_info["top_down"], rgb.shape[0])
                plot_frame = np.hstack((rgb, top_down))

            cv2.imshow(f"Scene {scene_id}, Episode: {episode_id}", \
                    plot_frame)
                
            keystroke = cv2.waitKey(0)
            while keystroke not in [ord(FORWARD_KEY), ord(LEFT_KEY), ord(RIGHT_KEY), ord(FINISH)]:
                print(f"INVALID KEY, Please try again.\n Valid Keys: [Forward: {FORWARD_KEY}, Left: {LEFT_KEY}, Right: {RIGHT_KEY}, Stop: {FINISH}]")
                keystroke = cv2.waitKey(0)
        
            if keystroke == ord(FORWARD_KEY):
                gui_action = HabitatSimActions.move_forward
                print("GUI action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                gui_action = HabitatSimActions.turn_left
                print("GUI action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                gui_action = HabitatSimActions.turn_right
                print("GUI action: RIGHT")
            elif keystroke == ord(FINISH):
                gui_action = HabitatSimActions.stop
                print("GUI action: FINISH")

                print("Stopping Run...")
            
            self.gui_action_t = torch.Tensor([gui_action]).unsqueeze(0).to(dtype=torch.int64)
            self._policy_info = self._get_policy_info(None)

            self._did_reset = False

            return PolicyActionData(
                actions=self.gui_action_t,
                rnn_hidden_states=rnn_hidden_states,
                policy_info=[self._policy_info],
            )

        if self.saved_nav:
            if not self._did_reset and masks[0] == 0: self._reset()
            self._done_initializing = True
            print(f"Inside saved nav condition for act")

        print("\n")
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _get_policy_info(self, detections):

        #Add step actions if gui_nav mode
        if self.gui_nav:
            policy_info = {}
            policy_info["step_action"] = {"step": int(self._num_steps - 1),
                                          "action": self.gui_action_t}

            return policy_info

        policy_info = super()._get_policy_info(detections) 
        # print(f"\n Current Agent Position: {policy_info['gps']}\n")

        #Saving Init Dictionary with initial position and relevant parameters
        init_dict = {}
        init_dict['grid_cell_size'] = self._obstacle_map.cell_size
        init_dict["map_shape"] = self._obstacle_map._map.shape
        init_dict["pixel_origin"] = [int(px) for px in self._obstacle_map._episode_pixel_origin]
        init_dict["pixels_per_meter"] = self._obstacle_map.pixels_per_meter
        policy_info["init_dict"] = init_dict   

        #Saving embedding dictionary with pixel grid points and corresponding embeddings
        policy_info["embed_dict"] = self._obstacle_map._embed_dict

        if not self._visualize:
            return policy_info
        
        #Visualize Embed Map
        robot_xy = self._observations_cache["robot_xy"].copy()
        policy_info["embed_map"] = self._obstacle_map.visualize_embed(center_pt=robot_xy)

        #Visualize Similarity Grid
        if not self.saved_nav:
            plot_point = self.top_xy if np.sum(self._last_frontier) == 0 else self._last_frontier
            if plot_point is not None:
                plot_point = plot_point[np.newaxis, :]
                plot_point = self._obstacle_map._xy_to_px(points = plot_point)
            
            policy_info["sim_grid"] = self.sim_grid.visualize(return_fig = True,
                                                            point_size=20,
                                                            plot_pts=plot_point)
        return policy_info

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:

        #Navigation using Saved Steps (Offline)
        if self.saved_nav:
            curr_step = int(self._num_steps)
            saved_action = self.step_actions[curr_step]
            print(f"Curr Step: {curr_step}, Saved Action: {saved_action}")
            return saved_action

        #Navigation using Frontiers (Online)
        frontiers = self._observations_cache["frontier_sensor"]

        #TODO: REMOVE
        #After 500 steps, or if no more frontiers, switch to navigate mode
        if self._num_steps > 400 or np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("\n\n------!! No frontiers found during exploration, navigating to the best grid...")

            self.nav_to_best_grid = True

            robot_xy = self._observations_cache["robot_xy"]
            self.xy_before_nav = robot_xy            
            goal = self._get_target_object_location(robot_xy)
            return self._pointnav(goal[:2], stop=True)
        
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)

        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        if pointnav_action[0] == 0: 
            print(f"Pointnav Action (during explore) is stop. Forcing Agent to continue exploration...")
            pointnav_action[0] = 1
        # print(f"PointNav Action: {pointnav_action}")
        return pointnav_action
    
    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
        use_sorted_items: Tuple[np.ndarray, np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Taken from BaseITMPolicy
        Returns the best frontier and its value based on closeby grid similarity scores.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        if use_sorted_items is not None:
            sorted_pts, sorted_values = use_sorted_items
        else:
            sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier

        return best_frontier, best_value

    def _sort_frontiers_by_value(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        # Check if the grid embedding associated to a frontier has a higher score

        ###----- Choosing Frontier using Nearest Grid points

        frontiers_px = self._obstacle_map._xy_to_px(points = frontiers)
        frontier_sim = {}
        
        for ind, f_px in enumerate(frontiers_px):

            dist_to_grids = {grid_px: np.linalg.norm(f_px - grid_px) for grid_px in self._obstacle_map._embed_dict[self._obstacle_map.curr_floor_name].keys()}
            
            # nearest_grid_px, _ = min(dist_to_grids.items(), key=lambda item: item[1])
            # nearest_grid_sim = self.sim_grid.sim_dict[0][nearest_grid_px]
            # frontier_sim[ind] = nearest_grid_sim

            nearest_grid_items = dict(heapq.nsmallest(3, dist_to_grids.items(), key = lambda item: item[1]))
            nearest_grid_sims = [self.sim_grid.sim_dict[0][grid_px] for grid_px in nearest_grid_items.keys()]
            nearest_sim = np.mean(nearest_grid_sims)
            frontier_sim[ind] = nearest_sim

        #Sort frontier indices by value, and then obtain the sorted frontiers
        sorted_items = dict(sorted(frontier_sim.items(), key=lambda item: item[1], reverse=True))
        sorted_frontiers = frontiers[list(sorted_items.keys())]
        sorted_values = np.array(list(sorted_items.values()))

        return sorted_frontiers, sorted_values
    

        ###----- Choosing Frontier using bounding box

        print(f"\n Choosing Frontier to explore...")
        frontiers_px = self._obstacle_map._xy_to_px(points = frontiers)
        # frontiers_px = np.array(frontiers_px)
        frontier_sim = {}

        #Define a bounding box for each frontier point, spaced out by the grid cell size in pixels.
        #Get all grid points lying inside this bounding box, and obtain their similarity values
        #Max of all obtained similarity values, and assign it to the current frontier point
        for ind, f_px in enumerate(frontiers_px):
            bound_box = np.array([[f_px - 0.6 * self._obstacle_map.cell_size],
                                  [f_px + 0.6 * self._obstacle_map.cell_size]])
            # print(bound_box.shape)
            bound_box = np.squeeze(bound_box, axis = 1)

            nearest_grid_pts = [grid_pt for grid_pt in self._obstacle_map._embed_dict[self._obstacle_map.curr_floor_name].keys() 
                                if self.is_point_in_box(pt = grid_pt, box_dims = bound_box)]

            #We obtain the processed (reduce by mean) similarity dictionary, which is present inside a dictionary with key 0
            max_sims = np.max([self.sim_grid.sim_dict[0][grid_pt] for grid_pt in nearest_grid_pts])

            frontier_sim[ind] = max_sims

        #Sort frontier indices by value, and then obtain the sorted frontiers
        sorted_items = dict(sorted(frontier_sim.items(), key=lambda item: item[1], reverse=True))
        sorted_frontiers = frontiers[list(sorted_items.keys())]
        sorted_values = np.array(list(sorted_items.values()))

        return sorted_frontiers, sorted_values

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        r"""
        Returns the goal position (in meters) if a grid similarity score is above a threshold.
        Else, return None (no goal yet) and keep exploring.
        """

        if self.saved_nav: return None


        text_prompt = self._target_object
        print(f"\n Text Prompt Used for Sim Grid: {text_prompt}")

        print(f" Number of Covered Grid Points: {len(self._obstacle_map._embed_dict[self._obstacle_map.curr_floor_name])}")

        start_time = time.time()

        self.sim_grid.load_sim_dict(embed_dict=self._obstacle_map._embed_dict[self._obstacle_map.curr_floor_name],
                                    text = text_prompt)

        #Get the top k prediction and similarity scores (sorted)
        top_pxs, top_sims = self.sim_grid.top_k_sims(k = self._top_k_preds, return_sims = True)
        
        time_taken = time.time() - start_time
        with open(save_times_path, 'a') as file:
            file.write(str(time_taken) + "\n")
        
        top_pxs, top_sims = top_pxs[0], top_sims[0]
        self.top_sims = top_sims

        print(f" Top Sim : {top_sims[0]}, Sim Threshold: {self.sim_goal_thresh}")

        ##-----No Goal Found: If similarity score is below threshold, return None
        if (top_sims[0] < self.sim_goal_thresh) and (not self.nav_to_best_grid): return None

        ##------If Goal is Found

        ###--- Treating Best pred as Goal

        # #Keeps track of whether agent is stuck in the same position or not
        # if self._count_same_pos < self._same_pos_thresh:
        #     if np.array_equal(position, self._prev_agent_xy): self._count_same_pos += 1
        #     else: self._count_same_pos = 0
        #     print(f"\n Count Same Position: {self._count_same_pos}\n")
        #     self._prev_agent_xy = position

        # #If agent is in same position for more than (say) 10 steps, then we assume that agent is in a cyclic frontier scenario
        # #We solve this by fixing our goal position to the current top grid pixel for the current and subsequent steps
        # if self._count_same_pos >= self._same_pos_thresh:
        #     print(f"Agent has been stuck in same position for at least {self._same_pos_thresh} steps. Fixing Goal position to : {self.top_xy}")
        #     return self.top_xy

        # top_px = top_px[0]
        # top_px = top_px[np.newaxis, :]
        # top_xy = self._obstacle_map._px_to_xy(px = top_px)
        # self.top_xy = top_xy.squeeze(0)

        # #Set Frontier to zero
        # self._last_frontier = np.zeros(2)

        # #Return Goal Position in meters
        # return self.top_xy

        ###--- Treating Preds as Frontiers

        #Filter and keep only the predictions within a 2 meter square distance from the top pred
        #The others are seen as invalid, since we are done exploring
        # mask_bound_box = np.array([[top_px[0] - 2 * self._obstacle_map.pixels_per_meter],
        #                            [top_px[0] + 2 * self._obstacle_map.pixels_per_meter]]).squeeze(1)

        # valid_inds = [ind for ind in range(len(top_px)) if self.is_point_in_box(pt = top_px[ind], box_dims = mask_bound_box)]
        # top_px = top_px[valid_inds]
        # top_sims = top_sims[valid_inds]
        

        # valid_top_px = [px for px in top_px if self.is_point_in_box(pt = px, box_dims = mask_bound_box)]
        # top_px = np.mean(valid_top_px, axis = 0)[np.newaxis, :]
        # print(f" Valid Top Grid Points: {len(valid_inds)} / {self._top_k_preds}")

        #Treat the top goal preds as frontiers to navigate towards
        top_xys = self._obstacle_map._px_to_xy(px = top_pxs)

        top_xy, top_sim = self._get_best_frontier(observations=None,
                                            frontiers=top_xys,
                                            use_sorted_items=(top_xys, top_sims))
        
        self.top_xy = top_xy
        

        #Return Goal Position in meters
        return self.top_xy
        
    def _get_object_detections(self, img):

        print('NO OBJECT DETECTIONS ARE BEING RUN')

        detections = vlm_obj_detection(
            image_source = img,
            boxes = torch.tensor([]),
            logits = torch.tensor([]),
            phrases = [],
            fmt='xyxy'
        )

        return detections

        # return super()._get_object_detections(img)

    def get_detection_position(self, ):
        pass

    @staticmethod
    def is_point_in_box(pt: np.ndarray, box_dims: np.ndarray) -> bool:
        r""""
        - pt: 2-dim coords of shape (1, 2)
        - box_dims: Top-left and bottom-right corners of the bounding box. Shape: (2, 2)        
        """
        # print(pt, box_dims)
        top_left, bottom_right = box_dims[0], box_dims[1]
        x_check = top_left[0] <= pt[0] <= bottom_right[0]
        y_check = top_left[1] <= pt[1] <= bottom_right[1]

        return (x_check and y_check)
    
    def _cache_observations(self: Union["HabitatMixin", BaseObjectNavPolicy], observations: TensorDict) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (TensorDict): The observations from the current timestep.
        """
        if len(self._observations_cache) > 0:
            return
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()

        #TODO: Need to add floor detection
        agent_height = observations["agent_height"][0].cpu()

        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        # Habitat GPS makes west negative, so flip y
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        self._obstacle_map: ObstacleMap
        if self._compute_frontiers:
            self._obstacle_map.update_map(
                rgb = rgb,                      
                agent_height = agent_height,     
                depth = depth,
                tf_camera_to_episodic=tf_camera_to_episodic,
                min_depth=self._min_depth,
                max_depth=self._max_depth,
                fx=self._fx,
                fy=self._fy,
                topdown_fov=self._camera_fov
            )
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        else:
            if "frontier_sensor" in observations:
                frontiers = observations["frontier_sensor"][0].cpu().numpy()
            else:
                frontiers = np.array([])

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": observations["depth"],  # for pointnav
            "robot_xy": robot_xy,
            "agent_height": agent_height,        #TODO: Need to add floor detection
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"][0].item(),
        }


@baseline_registry.register_policy
class HabitatGridPolicyV2(HabitatMixin, BaseObjectNavPolicy):
    r"Grid Policy with Image-Conditioned Detector"

    _count_same_pos: int = 0
    _prev_agent_xy: np.ndarray = np.zeros((2))
    _same_pos_thresh: int = 5
    _top_k_preds: int = 3
    nav_to_best_grid: bool = False
    xy_before_nav: Tuple[Any] = None
    stop_explore_after: int = 4000          #TODO Changed: Default 400        

    def __init__(self, 

        #Sim Grid Args
        prompt_mode: str = "text", 
        process_sim_mode: str = "mean",
        scrape_imgs: bool = False, 
        scrape_num: int = 3, 
        scrape_data_dir: str = "data/scraped_imgs/ovon_15",
        top_n_patch_embeds: int = 1,
        sim_goal_thresh: float = 0.75,

        *args, **kwargs):
            
            super().__init__(*args, **kwargs)
            self.gui_action_t = None
            self.sim_goal_thresh = sim_goal_thresh

            self.mode = "explore"
            self.goal_type = "feature_map"
            self.top_xy = []
            self.top_sims = []
            self._last_frontier, self._last_value = np.zeros(2), float("-inf")
            self._acyclic_enforcer = AcyclicEnforcer()

            self.owl_detector = OwlViT_Detector()
            self.detection_dist_thresh = 2          #TODO Changed: 1.5
            self.detection_score_thresh = 0.88      #TODO Changed: 0.95
            self.best_score = 0
            self.detection_pos = None

            if self._compute_frontiers:
                self.sim_grid = Sim_Grid_Online(model_name = self._obstacle_map.embed_model_name,
                                         use_model = self._obstacle_map._embed_model,
                                         grid_size = self._obstacle_map.cell_size,
                                         prompt_mode = prompt_mode,
                                         process_sim_mode = process_sim_mode,
                                         scrape_imgs = scrape_imgs,
                                         scrape_num = scrape_num,
                                         scrape_data_dir = scrape_data_dir,
                                         top_n_patch_embeds = top_n_patch_embeds)

    def _reset(self) -> None:

        super()._reset()

        self._count_same_pos = 0
        self._prev_agent_xy= np.zeros((2))
        self.nav_to_best_grid= False
        self.xy_before_nav = None

        self.mode = "explore"
        self.goal_type = "feature_map"
        self.top_xy = None
        self.top_sims = None
        self._last_frontier, self._last_value = np.zeros(2), float("-inf")
        self._acyclic_enforcer = AcyclicEnforcer()

        self.owl_detector.query_vector = None
        self.best_score = 0
        self.detection_pos = None


    def act(self,
        observations: TensorDict,
        rnn_hidden_states: Any,  # can be anything because it is not used
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
        gui_info: TensorDict = None,
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        
        #Navigation using GUI
        if self.gui_nav:
            if not self._did_reset and masks[0] == 0: self._reset()

            scene_id = gui_info["scene_id"]
            episode_id = gui_info["episode_id"]

            print(f"\nStep: {self._num_steps}")
            self._num_steps += 1

            if "top_down" not in gui_info:
                plot_frame = observations["rgb"][0].cpu().numpy().astype(np.uint8)[:, :, ::-1]                
            else:
                rgb = observations["rgb"][0].cpu().numpy().astype(np.uint8)[:, :, ::-1] 
                top_down = maps.colorize_draw_agent_and_fit_to_height(gui_info["top_down"], rgb.shape[0])
                plot_frame = np.hstack((rgb, top_down))

            cv2.imshow(f"Scene {scene_id}, Episode: {episode_id}", \
                    plot_frame)
                
            keystroke = cv2.waitKey(0)
            while keystroke not in [ord(FORWARD_KEY), ord(LEFT_KEY), ord(RIGHT_KEY), ord(FINISH)]:
                print(f"INVALID KEY, Please try again.\n Valid Keys: [Forward: {FORWARD_KEY}, Left: {LEFT_KEY}, Right: {RIGHT_KEY}, Stop: {FINISH}]")
                keystroke = cv2.waitKey(0)
        
            if keystroke == ord(FORWARD_KEY):
                gui_action = HabitatSimActions.move_forward
                print("GUI action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                gui_action = HabitatSimActions.turn_left
                print("GUI action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                gui_action = HabitatSimActions.turn_right
                print("GUI action: RIGHT")
            elif keystroke == ord(FINISH):
                gui_action = HabitatSimActions.stop
                print("GUI action: FINISH")

                print("Stopping Run...")
            
            self.gui_action_t = torch.Tensor([gui_action]).unsqueeze(0).to(dtype=torch.int64)
            self._policy_info = self._get_policy_info(None)

            self._did_reset = False

            return PolicyActionData(
                actions=self.gui_action_t,
                rnn_hidden_states=rnn_hidden_states,
                policy_info=[self._policy_info],
            )

        if self.saved_nav:
            if not self._did_reset and masks[0] == 0: self._reset()
            self._done_initializing = True
            print(f"Inside saved nav condition for act")

        print("\n")
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _get_policy_info(self, detections):

        #Add step actions if gui_nav mode
        if self.gui_nav:
            policy_info = {}
            policy_info["step_action"] = {"step": int(self._num_steps - 1),
                                          "action": self.gui_action_t}

            return policy_info

        policy_info = super()._get_policy_info(detections) 
        # print(f"\n Current Agent Position: {policy_info['gps']}\n")

        #Saving Init Dictionary with initial position and relevant parameters
        init_dict = {}
        init_dict['grid_cell_size'] = self._obstacle_map.cell_size
        init_dict["map_shape"] = self._obstacle_map._map.shape
        init_dict["pixel_origin"] = [int(px) for px in self._obstacle_map._episode_pixel_origin]
        init_dict["pixels_per_meter"] = self._obstacle_map.pixels_per_meter
        policy_info["init_dict"] = init_dict   

        #Current mode and type of pred
        policy_info["mode"] = self.mode
        policy_info["goal_type"] = self.goal_type

        #Saving embedding dictionary with pixel grid points and corresponding embeddings
        policy_info["embed_dict"] = self._obstacle_map._embed_dict

        if (not self._visualize) or (len(self._obstacle_map._embed_dict[self._obstacle_map.curr_floor_name]) == 0):
            return policy_info
        
        #Visualize Embed Map
        robot_xy = self._observations_cache["robot_xy"].copy()
        policy_info["embed_map"] = self._obstacle_map.visualize_embed(center_pt=robot_xy)

        #Visualize Similarity Grid
        if not self.saved_nav:
            plot_point = self.top_xy if np.sum(self._last_frontier) == 0 else self._last_frontier
            if plot_point is not None:
                plot_point = plot_point[np.newaxis, :]
                plot_point = self._obstacle_map._xy_to_px(points = plot_point)
            
            policy_info["sim_grid"] = self.sim_grid.visualize(return_fig = True,
                                                            point_size=20,
                                                            plot_pts=plot_point)
        return policy_info

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:

        #Navigation using Saved Steps (Offline)
        if self.saved_nav:
            curr_step = int(self._num_steps)
            saved_action = self.step_actions[curr_step]
            print(f"Curr Step: {curr_step}, Saved Action: {saved_action}")
            return saved_action

        #Navigation using Frontiers (Online)
        frontiers = self._observations_cache["frontier_sensor"]

        #TODO: REMOVE
        #After 500 steps, or if no more frontiers, switch to navigate mode
        if self._num_steps > self.stop_explore_after or np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("\n\n------!! No frontiers found during exploration, navigating to the best grid...")

            self.nav_to_best_grid = True

            robot_xy = self._observations_cache["robot_xy"]
            self.xy_before_nav = robot_xy            
            goal = self._get_target_object_location(robot_xy)
            return self._pointnav(goal[:2], stop=True)
        
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)

        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        if pointnav_action[0] == 0: 
            print(f"Pointnav Action (during explore) is stop. Forcing Agent to continue exploration...")
            pointnav_action[0] = 1
        # print(f"PointNav Action: {pointnav_action}")
        return pointnav_action
    
    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
        use_sorted_items: Tuple[np.ndarray, np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Taken from BaseITMPolicy
        Returns the best frontier and its value based on closeby grid similarity scores.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        if use_sorted_items is not None:
            sorted_pts, sorted_values = use_sorted_items
        else:
            sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("\n Sticking to last point.")
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier

        return best_frontier, best_value

    def _sort_frontiers_by_value(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        # Check if the grid embedding associated to a frontier has a higher score

        ###----- Choosing Frontier using Nearest Grid points

        frontiers_px = self._obstacle_map._xy_to_px(points = frontiers)
        frontier_sim = {}
        
        for ind, f_px in enumerate(frontiers_px):

            dist_to_grids = {grid_px: np.linalg.norm(f_px - grid_px) for grid_px in self._obstacle_map._embed_dict[self._obstacle_map.curr_floor_name].keys()}
            
            # nearest_grid_px, _ = min(dist_to_grids.items(), key=lambda item: item[1])
            # nearest_grid_sim = self.sim_grid.sim_dict[0][nearest_grid_px]
            # frontier_sim[ind] = nearest_grid_sim

            nearest_grid_items = dict(heapq.nsmallest(3, dist_to_grids.items(), key = lambda item: item[1]))
            nearest_grid_sims = [self.sim_grid.sim_dict[0][grid_px] for grid_px in nearest_grid_items.keys()]
            nearest_sim = np.mean(nearest_grid_sims)
            frontier_sim[ind] = nearest_sim

        #Sort frontier indices by value, and then obtain the sorted frontiers
        sorted_items = dict(sorted(frontier_sim.items(), key=lambda item: item[1], reverse=True))
        sorted_frontiers = frontiers[list(sorted_items.keys())]
        sorted_values = np.array(list(sorted_items.values()))

        return sorted_frontiers, sorted_values
    

        ###----- Choosing Frontier using bounding box

        print(f"\n Choosing Frontier to explore...")
        frontiers_px = self._obstacle_map._xy_to_px(points = frontiers)
        # frontiers_px = np.array(frontiers_px)
        frontier_sim = {}

        #Define a bounding box for each frontier point, spaced out by the grid cell size in pixels.
        #Get all grid points lying inside this bounding box, and obtain their similarity values
        #Max of all obtained similarity values, and assign it to the current frontier point
        for ind, f_px in enumerate(frontiers_px):
            bound_box = np.array([[f_px - 0.6 * self._obstacle_map.cell_size],
                                  [f_px + 0.6 * self._obstacle_map.cell_size]])
            # print(bound_box.shape)
            bound_box = np.squeeze(bound_box, axis = 1)

            nearest_grid_pts = [grid_pt for grid_pt in self._obstacle_map._embed_dict[self._obstacle_map.curr_floor_name].keys() 
                                if self.is_point_in_box(pt = grid_pt, box_dims = bound_box)]

            #We obtain the processed (reduce by mean) similarity dictionary, which is present inside a dictionary with key 0
            max_sims = np.max([self.sim_grid.sim_dict[0][grid_pt] for grid_pt in nearest_grid_pts])

            frontier_sim[ind] = max_sims

        #Sort frontier indices by value, and then obtain the sorted frontiers
        sorted_items = dict(sorted(frontier_sim.items(), key=lambda item: item[1], reverse=True))
        sorted_frontiers = frontiers[list(sorted_items.keys())]
        sorted_values = np.array(list(sorted_items.values()))

        return sorted_frontiers, sorted_values

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        r"""
        Returns the goal position (in meters) if a grid similarity score is above a threshold.
        Else, return None (no goal yet) and keep exploring.
        """

        if self.saved_nav: return None


        text_prompt = self._target_object
        print(f"\n Text Prompt Used for Sim Grid: {text_prompt}")
        print(f" Number of Covered Grid Points: {len(self._obstacle_map._embed_dict[self._obstacle_map.curr_floor_name])}")

        if len(self._obstacle_map._embed_dict[self._obstacle_map.curr_floor_name]) == 0:
            print(f"No grid points have been covered yet. Cannot compute similarity scores...")
            return None

        start_time = time.time()

        self.sim_grid.load_sim_dict(embed_dict=self._obstacle_map._embed_dict[self._obstacle_map.curr_floor_name],
                                    text = text_prompt)

        #Get the top k prediction and similarity scores (sorted)
        top_pxs, top_sims = self.sim_grid.top_k_sims(k = self._top_k_preds, return_sims = True)
        
        time_taken = time.time() - start_time
        with open(save_times_path, 'a') as file:
            file.write(str(time_taken) + "\n")
        
        top_pxs, top_sims = top_pxs[0], top_sims[0]        
        self.top_sims = top_sims

        print(f" Top Sim : {top_sims[0]}, Sim Threshold: {self.sim_goal_thresh}")

        ##-----No Goal Found: If similarity score is below threshold, return None
        if (top_sims[0] < self.sim_goal_thresh) and (not self.nav_to_best_grid): 
            self.mode = "explore"
            return None

        ##------If Goal is Found
        self.mode = "navigate"

        ###--- Treating Best pred as Goal

        # #Keeps track of whether agent is stuck in the same position or not
        # if self._count_same_pos < self._same_pos_thresh:
        #     if np.array_equal(position, self._prev_agent_xy): self._count_same_pos += 1
        #     else: self._count_same_pos = 0
        #     print(f"\n Count Same Position: {self._count_same_pos}\n")
        #     self._prev_agent_xy = position

        # #If agent is in same position for more than (say) 10 steps, then we assume that agent is in a cyclic frontier scenario
        # #We solve this by fixing our goal position to the current top grid pixel for the current and subsequent steps
        # if self._count_same_pos >= self._same_pos_thresh:
        #     print(f"Agent has been stuck in same position for at least {self._same_pos_thresh} steps. Fixing Goal position to : {self.top_xy}")
        #     return self.top_xy

        # top_px = top_px[0]
        # top_px = top_px[np.newaxis, :]
        # top_xy = self._obstacle_map._px_to_xy(px = top_px)
        # self.top_xy = top_xy.squeeze(0)

        # #Set Frontier to zero
        # self._last_frontier = np.zeros(2)

        # #Return Goal Position in meters
        # return self.top_xy

        ###--- Treating Preds as Frontiers

        #Filter and keep only the predictions within a 2 meter square distance from the top pred
        #The others are seen as invalid, since we are done exploring
        # mask_bound_box = np.array([[top_px[0] - 2 * self._obstacle_map.pixels_per_meter],
        #                            [top_px[0] + 2 * self._obstacle_map.pixels_per_meter]]).squeeze(1)

        # valid_inds = [ind for ind in range(len(top_px)) if self.is_point_in_box(pt = top_px[ind], box_dims = mask_bound_box)]
        # top_px = top_px[valid_inds]
        # top_sims = top_sims[valid_inds]
        

        # valid_top_px = [px for px in top_px if self.is_point_in_box(pt = px, box_dims = mask_bound_box)]
        # top_px = np.mean(valid_top_px, axis = 0)[np.newaxis, :]
        # print(f" Valid Top Grid Points: {len(valid_inds)} / {self._top_k_preds}")

        #Treat the top goal preds as frontiers to navigate towards
        top_xys = self._obstacle_map._px_to_xy(px = top_pxs)

        top_xy, top_sim = self._get_best_frontier(observations=None,
                                            frontiers=top_xys,
                                            use_sorted_items=(top_xys, top_sims))
        
        self.top_xy = top_xy


        # Check if target is detected by image-conditioned detector
        # Detection is run only in navigate mode, when agent is close to the target prediction
        obs = self._observations_cache["object_map_rgbd"][0]
        rgb_img, tf = obs[0], obs[2]
        curr_position = tf[:2, 3]
        dist_to_top_pred = np.linalg.norm(curr_position - top_xy)
        print(f" Dist to Top Pred: {dist_to_top_pred}")

        if dist_to_top_pred <= self.detection_dist_thresh:
            detect_score, detection_bbox = self._get_object_detections(rgb_img)

            #If detect_score is greater than thresh, then get position of detection, and use it as the goal
            # Change the detection goal only if the new detect_score is better than the old
            if detect_score >= self.detection_score_thresh:
                self.goal_type = "detection"
                detection_pos = self.get_detection_position(detection_bbox)

                #Sanity checks for detection position
                if detection_pos is None: 
                    print(f" No valid detection position found. Relying on top grid prediction...")
                    self.goal_type = "feature_map"
                    return self.top_xy

                if np.linalg.norm(detection_pos - top_xy) > self.detection_dist_thresh:
                    print(f" Detection position is too far away from top grid prediction. Relying on top grid prediction...")
                    self.goal_type = "feature_map"
                    return self.top_xy

                is_top_cyclic = self._acyclic_enforcer.check_cyclic(curr_position, top_xy, (detect_score, top_sim))
                is_detect_cyclic = self._acyclic_enforcer.check_cyclic(curr_position, detection_pos, (detect_score, top_sim))
                
                if is_top_cyclic and is_detect_cyclic:
                    print(f" Both top grid and detection positions are cyclic. Relying on top grid prediction...")
                    self.goal_type = "feature_map"
                    return self.top_xy
                elif is_detect_cyclic:
                    print(f" Suppressed cyclic detection position. Relying on top grid prediction...")
                    self.goal_type = "feature_map"
                    self._acyclic_enforcer.add_state_action(curr_position, top_xy, (detect_score, top_sim))
                    return self.top_xy

                #Change goal to detection position
                self.best_score = detect_score
                self.detection_pos = detection_pos
                self._acyclic_enforcer.add_state_action(curr_position, self.detection_pos, (detect_score, top_sim))
                print(f" Changing goal to detection position")

                return self.detection_pos
        
        #If detection fails, rely on the 
        #Return Goal Position in meters
        self.goal_type = "feature_map"
        return self.top_xy
        
    def _get_object_detections(self, img):

        score, target_bbox = self.owl_detector.is_query_in_image(target_image = img, return_box=True)
        xyxy_bbox = self.owl_detector.conv_to_xyxy_bbox(target_bbox)

        return score, xyxy_bbox

    def get_detection_position(self, detection_bbox):
        r"""
        Returns the position at which detection is located at
        """

        rgb, depth, tf, min_depth, max_depth, fx, fy = self._observations_cache["object_map_rgbd"][0]

        #Denorm detection bbox
        height, width = rgb.shape[:2]
        detection_bbox_denorm = detection_bbox * np.array([width, height, width, height])
        detection_bbox_denorm = list(detection_bbox_denorm)
        
        #Obtain mask for the detected object
        sam_object_mask = self._mobile_sam.segment_bbox(rgb, detection_bbox_denorm)
        object_mask = sam_object_mask.copy()
        object_mask[object_mask > 0] = 1

        #Get local and global cloud for the object
        local_cloud = self._object_map._extract_object_cloud(depth, object_mask, min_depth, max_depth, fx, fy)
        print(type(local_cloud), len(local_cloud), np.shape(local_cloud))
        if len(local_cloud) == 0: return None
        global_cloud = transform_points(tf, local_cloud)

        #Get closest point of the object cloud wrt current position
        curr_position = tf[:3, 3]
        closest_point = self._object_map._get_closest_point(global_cloud, curr_position)

        return closest_point[:2]

    def _update_object_map(self, rgb, depth, tf_camera_to_episodic, min_depth, max_depth, fx, fy):
        
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)

        detections = ObjectDetections(
            image_source = rgb,
            boxes = torch.tensor([]),
            logits = torch.tensor([]),
            phrases = [],
            fmt='xyxy'
        )

        #cone_fov = get_fov(fx, depth.shape[1])
        #self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        return detections


    @staticmethod
    def is_point_in_box(pt: np.ndarray, box_dims: np.ndarray) -> bool:
        r""""
        - pt: 2-dim coords of shape (1, 2)
        - box_dims: Top-left and bottom-right corners of the bounding box. Shape: (2, 2)        
        """
        # print(pt, box_dims)
        top_left, bottom_right = box_dims[0], box_dims[1]
        x_check = top_left[0] <= pt[0] <= bottom_right[0]
        y_check = top_left[1] <= pt[1] <= bottom_right[1]

        return (x_check and y_check)
    
    def _cache_observations(self: Union["HabitatMixin", BaseObjectNavPolicy], observations: TensorDict) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (TensorDict): The observations from the current timestep.
        """
        if len(self._observations_cache) > 0:
            return

        if (self.owl_detector.query_vector is None) and (not self.saved_nav) and (not self.gui_nav):
            
            #Query Vector only corresponds to the top-most result
            # print(f"Obtaining Top Query Vector...")
            # source_image = load_images(query = self._target_object, 
            #                            num_images = 1, 
            #                            save_dir = self.sim_grid.scrape_data_dir)[0]

            # self.owl_detector.extract_query_vector(texts = [self._target_object],
            #                                        source_image = source_image,
            #                                        with_objectness = False)

            #Query Vector is an average of the top-k results
            print(f"Obtaining Mean Query Vector...")
            query_images = load_images(query = self._target_object, 
                                       num_images = self.sim_grid.scrape_num, 
                                       save_dir = self.sim_grid.scrape_data_dir)

            query_vecs = []
            for query_img in query_images:

                self.owl_detector.extract_query_vector(texts = [self._target_object],
                                    source_image = query_img,
                                    with_objectness = False)
                query_vecs.append(self.owl_detector.query_vector)

            query_vec = torch.stack(query_vecs, dim=0).mean(dim=0)
            self.owl_detector.query_vector = query_vec                                    

            print(f" Query Vector for OWL-ViT Detector is initialized!\n\n")
    
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()

        #TODO: Need to add floor detection
        agent_height = observations["agent_height"][0].cpu()

        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        # Habitat GPS makes west negative, so flip y
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        self._obstacle_map: ObstacleMap
        if self._compute_frontiers:
            self._obstacle_map.update_map(
                rgb = rgb,                      
                agent_height = agent_height,     
                depth = depth,
                tf_camera_to_episodic=tf_camera_to_episodic,
                min_depth=self._min_depth,
                max_depth=self._max_depth,
                fx=self._fx,
                fy=self._fy,
                topdown_fov=self._camera_fov
            )
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        else:
            if "frontier_sensor" in observations:
                frontiers = observations["frontier_sensor"][0].cpu().numpy()
            else:
                frontiers = np.array([])

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": observations["depth"],  # for pointnav
            "robot_xy": robot_xy,
            "agent_height": agent_height,        #TODO: Need to add floor detection
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"][0].item(),
        }


cs = ConfigStore.instance()
cs.store(group="habitat_baselines/rl/policy", name="vlfm_policy", node=VLFMPolicyConfig)
