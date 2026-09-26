"""Generate HSSD rare-category episodes by distributing each category's target quota 
(e.g. 50 episodes per category) across eligible scenes in proportion to their subcategory 
counts, then using later rounds to redistribute any remaining category deficits. 
Load or resume each scene, generate object viewpoints, sample valid starts by path 
distance and floor consistency, and save the resulting episodes and diagnostics.

After generation, check the logs to verify category counts and review failed object
or viewpoint generation before filtering the reference inventory for future runs.
"""

import argparse
from collections import Counter
import gzip
import json
import os
import random
from typing import Dict, Sequence, Tuple, Union

import habitat_sim
from habitat_sim.simulator import Simulator
from habitat_sim.utils import common as utils
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.generate_viewpoints import ViewPoints_Generator

SAVE_DIR = "/mnt/vlfm_query_embed/dataset/hssd_rare_eps"
DATA_SPLIT = "val"

SCENE_DATASET_ROOT_DIR = (
    "/mnt/vlfm_query_embed/habitat-lab/data/scene_datasets/hssd-hab"
)
DATASET_ROOT_DIR = (
    "/mnt/vlfm_query_embed/habitat-lab/data/datasets/objectnav/hssd/val_rare"
)

REF_SCENE_INST = "/mnt/vlfm_query_embed/dataset/main_cat_scene_inst.json"
REF_CAT_SCENE_INST_COUNTS = (
    "/mnt/vlfm_query_embed/dataset/main_cat_scene_inst_counts_filt.json"
)
REF_CAT_SPLITS = "/mnt/vlfm_query_embed/dataset/main_cat_split.json"

os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"


class HSSD_Rare_Generator:
    """Build subcategory goals and fill main-category episode quotas.

    Retry budgets have separate scopes: start_sample_attempts bounds proposals
    per sampling call, candidate_retries bounds failed calls per subcategory in
    one scene pass, and generation_rounds bounds passes over eligible scenes.
    """

    def __init__(
        self,
        num_eps_per_cat: int,
        agent_height: float,
        agent_radius: float,
        sensor_height: float,
        hfov: float,
        img_size: Tuple[int, int],
        scene_dataset_dir=SCENE_DATASET_ROOT_DIR,
        dataset_root_dir=DATASET_ROOT_DIR,
        save_dir: str = SAVE_DIR,
        data_split: str = DATA_SPLIT,
        cat_scene_inst_counts_path: str = REF_CAT_SCENE_INST_COUNTS,
        cat_scene_inst_path: str = REF_SCENE_INST,
        cat_splits_path: str = REF_CAT_SPLITS,
        num_incr_steps: int = 3,
        start_pos_min_radius: float = 3.0,
        start_pos_max_radius: float = 10.0,
        eps_per_obj: int = 1,
        floor_thresh: float = 0.25,
        verbose: bool = False,
        start_sample_attempts: int = 100,
        candidate_retries: int = 3,
        generation_rounds: int = 3,
    ):

        if (
            num_eps_per_cat < 1
            or min(start_sample_attempts, candidate_retries, generation_rounds) < 1
        ):
            raise ValueError("Episode quota and retry budgets must be positive")
        if not 0 <= start_pos_min_radius < start_pos_max_radius:
            raise ValueError(
                "Require 0 <= minimum goal distance < maximum goal distance"
            )
        self.num_eps_per_cat = num_eps_per_cat
        self.start_sample_attempts = start_sample_attempts
        self.candidate_retries = candidate_retries
        self.generation_rounds = generation_rounds
        self.subcategory_counts = Counter()
        os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)

        self.agent_height = agent_height
        self.agent_radius = agent_radius
        self.sensor_height = sensor_height
        self.hfov = hfov
        self.img_size = img_size

        self.scene_dataset_dir = scene_dataset_dir
        self.dataset_path = dataset_root_dir
        self.save_dir = save_dir
        self.split = data_split

        self.cat_scene_inst_counts_path = cat_scene_inst_counts_path
        self.cat_scene_inst_path = cat_scene_inst_path
        self.cat_splits_path = cat_splits_path

        self.cat_scene_inst_counts_filt_path = os.path.join(
            save_dir, "logs", "main_cat_scene_inst_counts_filt.json"
        )
        if not os.path.exists(self.cat_scene_inst_counts_filt_path):
            self.save_json(
                self.cat_scene_inst_counts_filt_path,
                self.open_json(self.cat_scene_inst_counts_path),
            )

        self.start_pos_min_radius = start_pos_min_radius
        self.start_pos_max_radius = start_pos_max_radius
        self.num_incr_steps = num_incr_steps
        self.eps_per_obj = eps_per_obj
        self.floor_thresh = floor_thresh
        self.verbose = verbose

        self.sim = None
        self.goals_by_category = {}
        self.episodes = []

        self.scene_names = self.all_scene_names()
        self.objects_info = self.load_objects_info()
        self.val_rare = self.load_val_rare()

        # Historical goal-failure log, retained for compatibility. It is not an
        # episode blacklist: a failed random start does not invalidate an object.
        self.ignore_obj_insts = self.load_ignored_obj_insts()

        # Diagnostics for the current run; scene files hold resumable progress.
        self.log_info = {"goals_by_category": {}, "episodes": {}}
        self.checkpoint = None

    def _load_sim(self, scene: str) -> None:

        if self.sim is not None:
            self.sim.close()

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = os.path.join(
            self.scene_dataset_dir, "hssd-hab.scene_dataset_config.json"
        )
        sim_cfg.scene_id = scene

        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.allow_sliding = False

        sensor_specs = []
        for name, sensor_type in (
            ("color", habitat_sim.SensorType.COLOR),
            ("depth", habitat_sim.SensorType.DEPTH),
        ):
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = f"{name}_sensor"
            sensor_spec.sensor_type = sensor_type
            sensor_spec.resolution = [self.img_size[0], self.img_size[1]]
            sensor_spec.position = [0.0, self.sensor_height, 0.0]
            sensor_spec.hfov = self.hfov
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(sensor_spec)

        # create agent specifications
        agent_cfg = habitat_sim.agent.AgentConfiguration(
            height=self.agent_height,
            radius=self.agent_radius,
            sensor_specifications=sensor_specs,
            action_space={
                "move_forward": habitat_sim.agent.ActionSpec(
                    "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
                ),
                "turn_left": habitat_sim.agent.ActionSpec(
                    "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
                ),
                "turn_right": habitat_sim.agent.ActionSpec(
                    "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
                ),
            },
        )

        self.sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [agent_cfg])
        )

        # set the navmesh
        print(f"Pathfinder is Loaded: {self.sim.pathfinder.is_loaded}")
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = self.agent_height
        navmesh_settings.agent_radius = self.agent_radius
        navmesh_settings.agent_max_climb = 0.2
        navmesh_settings.cell_height = 0.2
        navmesh_settings.include_static_objects = True
        navmesh_success = self.sim.recompute_navmesh(
            self.sim.pathfinder, navmesh_settings
        )
        print(f"Navmesh Recompute Success: {navmesh_success}")
        assert navmesh_success, "Failed to build the navmesh!"

    def all_scene_names(self, split=None):
        r"""
        If split is None, loads all scene names.
        If split is given (train/val), loads only corresponding scene names.
        """

        if split is not None:
            base_dir = os.path.dirname(self.dataset_path)
            scenes_dir = os.path.join(base_dir, f"{split}/content")
            assert os.path.exists(
                scenes_dir
            ), f"Dataset path does not exist for split: {split}! Please provide a valid split value."

        else:
            scenes_dir = os.path.join(self.scene_dataset_dir, "scenes")

        names = [file_name.split(".")[0] for file_name in os.listdir(scenes_dir)]
        return names

    def load_scene_info(self, scene_id):
        scene_info_path = os.path.join(
            self.scene_dataset_dir, f"scenes/{scene_id}.scene_instance.json"
        )

        with open(scene_info_path, "r") as f:
            return json.load(f)

    def load_objects_info(self):

        def get_rare_cat(main_cat, descr):

            if not isinstance(main_cat, str) or (main_cat == "nan"):
                return descr
            return f"{main_cat}, {descr}"

        obj_path = os.path.join(self.scene_dataset_dir, "semantics/rare_objects.csv")
        if os.path.exists(obj_path):
            return pd.read_csv(obj_path, index_col=0)

        # If file not found, create the file
        # Process:
        #   - Filter to id, name, main_category
        #   - Create new column, rare_category, by appending main_category and name
        #   - Save file
        print(f"\nObject Info File not Found! Creating the file...")
        obj_info_path = os.path.join(self.scene_dataset_dir, "semantics/objects.csv")
        assert os.path.exists(obj_info_path), "Not able to find objects.csv!"

        df = pd.read_csv(obj_info_path)
        df_filt = df[["id", "name", "main_category", "aligned.dims"]].copy()
        df_filt = df_filt.dropna(subset=["name"])

        # Add category column: main_category + name
        df_filt["rare_category"] = df_filt.apply(
            lambda row: get_rare_cat(row["main_category"], row["name"]), axis=1
        )

        # Filter rows with all NaN values
        df_filt = df_filt.dropna(how="all", ignore_index=False)

        # Save to path
        df_filt.to_csv(obj_path)
        print(f"Saved Objects Info to {obj_path}")

        return df_filt

    def load_val_rare(self):
        val_rare_path = os.path.join(self.dataset_path, "val_rare.json.gz")

        if os.path.exists(val_rare_path):
            with gzip.open(val_rare_path, "rt") as f:
                return json.load(f)

        # If file does not exist, create the file
        print(f"\nval_rare.json.gz not Found! Creating the file...")
        val_rare = {
            "episodes": [],
            "category_to_task_category_id": {},
            "category_to_scene_annotation_category_id": {},
        }

        ind_to_cat = self.objects_info["rare_category"].to_dict()
        cat_to_ind = {v: k for k, v in ind_to_cat.items()}

        val_rare["category_to_task_category_id"] = cat_to_ind
        val_rare["category_to_scene_annotation_category_id"] = cat_to_ind

        # Save to path
        self.zip_and_save(val_rare, val_rare_path)

        return val_rare

    def obj_id_from_name(self, obj_name):
        return self.objects_info[self.objects_info.id == obj_name].index.item()

    def obj_category_from_name(self, obj_name):

        return self.objects_info[self.objects_info.id == obj_name].rare_category.item()

    def obj_dims_from_name(self, obj_name):

        s_dims = self.objects_info[self.objects_info.id == obj_name][
            "aligned.dims"
        ].item()
        return [float(elem) for elem in s_dims.split(",")]

    def generate_goals_by_category(self, scene: str):

        self.checkpoint = self.load_checkpoint(scene)

        print(f"Loading Simulator for scene: {scene}")
        self._load_sim(scene)
        view_pts_gen = ViewPoints_Generator(
            sim=self.sim,
            scene_id=scene,
            scene_dataset_path=self.scene_dataset_dir,
        )

        self.goals_by_category = {}
        scene_info = self.load_scene_info(scene)
        scene_objects = scene_info["object_instances"]

        # Load reference instances for the current scene
        ref_insts = self.load_ref_scene_eps(scene)
        total_main_cats = len(ref_insts)

        print(f"\n\n---- Generating goals_by_category for Scene: {scene} ----")
        for done_ind, scene_cat_info in enumerate(ref_insts):

            _, obj_main_cat, obj_names = scene_cat_info
            print(
                f"\n - Main Category: {obj_main_cat}. Categories Done: {done_ind}/{total_main_cats}"
            )

            # Generates goals_by_category info for each object subcategories
            for obj_name in tqdm(obj_names):

                tracker_obj_invalid = 0
                tracker_view_pts = 0
                tracker_obj_valid = 0

                # For each object, obtain all the instances in the current scene
                # Then, generate info centered on each of those instances

                scene_obj_info = [
                    obj for obj in scene_objects if obj["template_name"] == obj_name
                ]
                num_obj_insts = len(scene_obj_info)
                assert (
                    num_obj_insts > 0
                ), f"No instances found in scene {scene} for object id : {obj_name}"

                # Obtain object info
                obj_id = self.obj_id_from_name(obj_name)
                obj_category = self.obj_category_from_name(obj_name)
                obj_dims = self.obj_dims_from_name(obj_name)

                key = f"{scene}_{obj_category}"
                print(
                    f"\nObject Category: {obj_category}, Object Name: {obj_name}, Instances: {num_obj_insts}"
                )

                # For each object instance, generate info
                for scene_inst_info in scene_obj_info:

                    obj_inst_position = scene_inst_info["translation"]

                    # Check if saved in checkpoint
                    if key in self.checkpoint["goals_by_category"]:

                        # Define a new list for (obj, scene)
                        if key not in self.goals_by_category:
                            self.goals_by_category[key] = []

                        # Add saved object info
                        skip_iter = False
                        for saved_obj in self.checkpoint["goals_by_category"][key]:
                            if saved_obj["position"] == obj_inst_position:
                                saved_obj["object_name"] = (
                                    f"{obj_name}_:{self.padded_int(num = len(self.goals_by_category[key]))}"
                                )
                                self.goals_by_category[key].append(saved_obj)

                                skip_iter = True
                                break

                        if skip_iter:
                            continue

                    # If the object doesn't exist in the source objects_info, skip
                    if (obj_id is None) or (obj_category is None):
                        tracker_obj_invalid += 1
                        self.ignore_obj_insts.add(
                            (scene, obj_main_cat, obj_name, tuple(obj_inst_position))
                        )
                        continue

                    # Generate Object Viewpoints with position and rotation to face the object
                    view_pts = view_pts_gen.generate_view_pts(
                        obj_pos=obj_inst_position,
                        obj_dims=obj_dims,
                        boundary_pts_dist=0.5,
                        boundary_check_radius=2,
                        dist_btw_view_pts=0.2,
                        view_pts_max_radius=1,
                    )

                    if len(view_pts) == 0:
                        tracker_view_pts += 1
                        self.ignore_obj_insts.add(
                            (scene, obj_main_cat, obj_name, tuple(obj_inst_position))
                        )
                        continue

                    # Define a new list for (obj, scene)
                    if key not in self.goals_by_category:
                        self.goals_by_category[key] = []

                    curr_obj_info = {}
                    curr_obj_info["position"] = obj_inst_position
                    curr_obj_info["object_id"] = obj_id
                    curr_obj_info["object_name"] = (
                        f"{obj_name}_:{self.padded_int(num = len(self.goals_by_category[key]))}"
                    )
                    curr_obj_info["object_category"] = obj_category

                    curr_obj_info["view_points"] = [
                        {
                            "agent_state": {
                                "position": [
                                    elem.astype(np.float64) for elem in view_pt[0]
                                ],
                                "rotation": view_pt[1].astype(np.float64).tolist(),
                            },
                            "iou": 0.0,
                        }
                        for view_pt in view_pts
                    ]

                    self.goals_by_category[key].append(curr_obj_info)

                    tracker_obj_valid += 1

                    # Save Checkpoint
                    self.save_checkpoint(scene)

                # Logging Info

                if obj_main_cat not in self.log_info["goals_by_category"]:
                    self.log_info["goals_by_category"][obj_main_cat] = {}

                if scene not in self.log_info["goals_by_category"][obj_main_cat]:
                    self.log_info["goals_by_category"][obj_main_cat][scene] = {}

                if (
                    obj_name
                    not in self.log_info["goals_by_category"][obj_main_cat][scene]
                ):
                    self.log_info["goals_by_category"][obj_main_cat][scene][
                        obj_name
                    ] = {}

                    self.log_info["goals_by_category"][obj_main_cat][scene][obj_name][
                        "failure_obj_invalid"
                    ] = 0
                    self.log_info["goals_by_category"][obj_main_cat][scene][obj_name][
                        "failure_view_pts"
                    ] = 0
                    self.log_info["goals_by_category"][obj_main_cat][scene][obj_name][
                        "success_obj_goals"
                    ] = 0

                self.log_info["goals_by_category"][obj_main_cat][scene][obj_name][
                    "failure_obj_invalid"
                ] += tracker_obj_invalid
                self.log_info["goals_by_category"][obj_main_cat][scene][obj_name][
                    "failure_view_pts"
                ] += tracker_view_pts
                self.log_info["goals_by_category"][obj_main_cat][scene][obj_name][
                    "success_obj_goals"
                ] += tracker_obj_valid

    def sample_start_for_goals(self, goals):
        """Validate the actual shortest goal path, with distances measured in meters.

        Both distance limits are geodesic distances to the nearest category
        viewpoint. The chosen shortest path must stay within floor_thresh.
        A random viewpoint anchors each proposal on a goal navigation island.
        Returns (sample, rejection_counts), with sample=None on exhaustion.
        Euclidean distance is to the chosen path endpoint, not the object center.
        """
        endpoints = np.asarray(
            [
                vp["agent_state"]["position"]
                for goal in goals
                for vp in goal["view_points"]
            ],
            dtype=np.float32,
        )
        if endpoints.size == 0:
            return None, {"no_viewpoints": 1}
        endpoints = endpoints.reshape(-1, 3)
        endpoints = np.asarray(
            [
                pt
                for pt in endpoints
                if np.all(np.isfinite(pt)) and self.sim.pathfinder.is_navigable(pt)
            ]
        )
        if not len(endpoints):
            return None, {"no_navigable_viewpoints": 1}
        failures = Counter()
        for _ in range(self.start_sample_attempts):
            anchor = endpoints[np.random.randint(len(endpoints))]
            island = self.sim.pathfinder.get_island(anchor)
            start = self.sim.pathfinder.get_random_navigable_point_near(
                circle_center=anchor,
                radius=self.start_pos_max_radius,
                island_index=island,
            )
            if start is None or not np.all(np.isfinite(start)):
                failures["invalid_start"] += 1
                continue
            if not self.sim.pathfinder.is_navigable(start):
                failures["invalid_start"] += 1
                continue
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = np.asarray(start, dtype=np.float32)
            path.requested_ends = endpoints
            if not self.sim.pathfinder.find_path(path) or not np.isfinite(
                path.geodesic_distance
            ):
                failures["no_path"] += 1
                continue
            if (
                not self.start_pos_min_radius
                <= path.geodesic_distance
                <= self.start_pos_max_radius
            ):
                failures["goal_distance"] += 1
                continue
            points = np.asarray(path.points)
            if (
                not len(points)
                or not np.all(np.isfinite(points))
                or np.ptp(points[:, 1]) > self.floor_thresh
            ):
                failures["different_floor"] += 1
                continue
            return {
                "start_position": np.asarray(start).tolist(),
                "geodesic_distance": float(path.geodesic_distance),
                "euclidean_distance": float(np.linalg.norm(start - points[-1])),
                "goal_viewpoint_position": points[-1].tolist(),
            }, dict(failures)
        return None, dict(failures)

    def generate_episodes(self, scene: str, quotas=None):
        """Generate episodes for the given scene, using the provided quotas or loading them from the reference file.
        If some episodes are already generated, keep them as is, and append to the set to satisfy the required number. 
        Stop after controlled retry limits, and also favor the underrepresented object subcategories when sampling.

        Failure here is transient: it never blacklists an object or changes the
        reference inventory. Only objects with saved/generated goals participate.
        """
        if quotas is None:
            quotas = {cat: n for n, cat, _ in self.load_ref_scene_eps(scene)}
        if self.episodes is None:
            self.episodes = []
        refs = self.open_json(self.cat_scene_inst_path)
        for main_cat, requested in quotas.items():
            templates = set(refs.get(main_cat, {}).get(scene, {}))
            candidates = {}
            for goals in self.goals_by_category.values():
                for goal in goals:
                    template = goal["object_name"].split("_:")[0]
                    if template in templates and goal.get("view_points"):
                        candidates.setdefault(goal["object_category"], []).append(goal)
            scene_log = (
                self.log_info["episodes"].setdefault(main_cat, {}).setdefault(scene, {})
            )
            summary = scene_log.setdefault(
                "sampling_summary",
                {"requested": 0, "generated": 0, "no_goal_candidates": 0},
            )
            summary["requested"] += requested
            if not candidates:
                summary["no_goal_candidates"] += requested
            # Count exhausted sampling calls, not individual rejected starts.
            failures = Counter()
            made = 0
            while made < requested:
                available = [
                    cat for cat in candidates if failures[cat] < self.candidate_retries
                ]
                if not available:
                    break
                # Counts include other scenes and previous runs. Randomize ties
                # so equally represented subcategories have equal priority.
                random.shuffle(available)
                category = min(available, key=lambda cat: self.subcategory_counts[cat])
                goals = candidates[category]
                sample, reasons = self.sample_start_for_goals(goals)
                log = (
                    self.log_info["episodes"]
                    .setdefault(main_cat, {})
                    .setdefault(scene, {})
                    .setdefault(category, {"attempts": 0, "success": 0, "failures": {}})
                )
                log["attempts"] += 1
                for reason, count in reasons.items():
                    log["failures"][reason] = log["failures"].get(reason, 0) + count
                if sample is None:
                    failures[category] += 1
                    continue
                angle = np.random.uniform(0, 2 * np.pi)
                self.episodes.append(
                    {
                        "episode_id": len(self.episodes),
                        "scene_id": scene,
                        "scene_dataset_config": os.path.join(
                            self.scene_dataset_dir, "hssd-hab.scene_dataset_config.json"
                        ),
                        "object_category": category,
                        "goals": [],
                        "start_position": sample.pop("start_position"),
                        "start_rotation": [
                            0,
                            float(np.sin(angle / 2)),
                            0,
                            float(np.cos(angle / 2)),
                        ],
                        "info": sample,
                    }
                )
                made += 1
                self.subcategory_counts[category] += 1
                log["success"] += 1
            summary["generated"] += made
            print(f"{scene}: {main_cat}: generated {made}/{requested}")

    def generate_scene_ep_counts(self, remove_elems=None):
        """Write the initial plan; optional removals edit the legacy local inventory.

        Runtime sampling failures never call this method or remove eligibility.
        """
        if remove_elems:
            cat_scene_inst_counts = self.open_json(self.cat_scene_inst_counts_filt_path)

            for elem in remove_elems:

                cat_name, scene_name = elem[0], elem[1]
                assert (
                    cat_name in cat_scene_inst_counts
                ), f"Category {cat_name} not found in category-scene instance counts!"
                assert (
                    scene_name in cat_scene_inst_counts[cat_name]
                ), f"Scene {scene_name} not found in category-scene instance counts for category {cat_name}!"

                print(f"Removing {elem} from category-scene instance counts...")
                del cat_scene_inst_counts[cat_name][scene_name]

            self.save_json(self.cat_scene_inst_counts_filt_path, cat_scene_inst_counts)

        print(f"Generating scene episode counts...")
        self.gen_eps_counts_per_scene()

    def gen_eps_counts_per_scene(self, return_out=False):
        """Apportion the quota by instance counts using largest remainders.

        Rows are [preferred_count, main_category, template_ids]. Zero-count
        rows remain available as fallback scenes during deficit redistribution.
        """
        cat_scene_inst = self.open_json(self.cat_scene_inst_path)
        cat_scene_inst_counts = self.open_json(self.cat_scene_inst_counts_filt_path)
        valid_cats = self.valid_categories()

        cat_scene_eps = {}
        for cat in tqdm(valid_cats):
            cat_scene_eps[cat] = {}

            weights = cat_scene_inst_counts.get(cat, {})
            weights = {scene: count for scene, count in weights.items() if count > 0}
            total = sum(weights.values())
            if not total:
                continue
            # Largest remainders: fixed denominator and quota, independent of
            # insertion order. Empty categories are reported by the final audit.
            exact = {
                scene: self.num_eps_per_cat * weight / total
                for scene, weight in weights.items()
            }
            counts = {scene: int(value) for scene, value in exact.items()}
            remainder = self.num_eps_per_cat - sum(counts.values())
            order = sorted(
                weights, key=lambda scene: (-(exact[scene] - counts[scene]), scene)
            )
            for scene in order[:remainder]:
                counts[scene] += 1
            for scene, count in counts.items():
                cat_scene_eps[cat][scene] = [
                    count,
                    cat,
                    list(cat_scene_inst[cat][scene]),
                ]

        # --- Create dict with Scene -> List of Eps ----
        scene_eps = {}
        for cat in cat_scene_eps:
            for scene in cat_scene_eps[cat]:

                add_insts = cat_scene_eps[cat][scene]

                if scene not in scene_eps:
                    scene_eps[scene] = []

                scene_eps[scene].append(add_insts)

        # --- Save scene_eps ---
        save_path = os.path.join(self.save_dir, "logs", f"scene_ep_counts.json")
        self.save_json(save_path, scene_eps)

        if return_out:
            return scene_eps, cat_scene_eps

    def load_ref_scene_eps(self, scene: str):
        r"""
        Loads reference scene episode counts and object names for the given scene.
        """

        ref_scene_eps = self.open_json(
            os.path.join(self.save_dir, "logs", f"scene_ep_counts.json")
        )
        return ref_scene_eps[scene]

    def load_ignored_obj_insts(self):
        r"""
        Loads the set of ignored object instances for reference.
        """

        empty_ignore_set = set()

        save_path = os.path.join(self.save_dir, "logs", "ignored_obj_insts.json")
        if not os.path.exists(save_path):
            print(
                f"No saved Ignored Object Instances found at {save_path}. Starting with an empty set."
            )
            return empty_ignore_set

        with open(save_path, "r") as f:
            info = json.load(f)

        print(
            f"Loaded Ignored Object Instances from {save_path}. Number of ignored instances: {len(info['ignore_obj_insts'])}"
        )
        return set(
            (elem[0], elem[1], elem[2], tuple(elem[3]))
            for elem in info["ignore_obj_insts"]
        )

    def save_ignored_obj_insts(self):
        r"""
        Saves the set of ignored object instances for reference.
        """

        info = {}
        info["ignore_obj_insts"] = list(self.ignore_obj_insts)

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, "logs", f"ignored_obj_insts.json")

        with open(save_path, "w") as f:
            json.dump(info, f, indent=4)

    def load_checkpoint(self, scene: str):
        r"""
        Open Checkpoint file for reference.
        """

        empty_checkpoint = {"goals_by_category": {}}

        save_path = os.path.join(self.save_dir, "logs", "checkpoint.json")
        if not os.path.exists(save_path):
            return empty_checkpoint

        with open(save_path, "r") as f:
            info = json.load(f)

        if info["scene"] == scene:
            return info
        return empty_checkpoint

    def save_checkpoint(self, scene: str):
        r"""
        Saves Checkpoint for reference.
        """

        info = {}
        info["scene"] = scene
        info["goals_by_category"] = self.goals_by_category

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, "logs", f"checkpoint.json")

        with open(save_path, "w") as f:
            json.dump(info, f)

        self.save_ignored_obj_insts()

    def _generate_and_save_scene(self, scene, scenes, quotas):
        """Reuse saved goals, append episodes, and persist this scene's progress."""
        if scene in scenes:
            self._load_sim(scene)
            self.goals_by_category = scenes[scene]["goals_by_category"]
            self.episodes = scenes[scene]["episodes"]
        else:
            self.generate_goals_by_category(scene)
            self.episodes = []
        self.generate_episodes(scene, quotas)
        data = scenes.get(
            scene,
            {
                "category_to_task_category_id": self.val_rare[
                    "category_to_task_category_id"
                ],
                "category_to_scene_annotation_category_id": self.val_rare[
                    "category_to_scene_annotation_category_id"
                ],
            },
        )
        data.update(goals_by_category=self.goals_by_category, episodes=self.episodes)
        scenes[scene] = data
        self.write_scene(scene, data)
        self.zip_and_save(
            self.log_info,
            os.path.join(self.save_dir, "logs", "meta_log.json.gz"),
            overwrite=True,
        )

    def _close_sim(self):
        """Release rendering resources after either generation entry point."""
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def save_objectnav_dataset_for_scene(self, scene: str):
        """Top up one scene to its plan; this is not a full-dataset completion run."""
        scenes = self.read_saved_scenes()
        self.subcategory_counts = Counter(
            ep["object_category"] for data in scenes.values() for ep in data["episodes"]
        )
        counts = self.episode_counts(scenes)
        existing = Counter(
            self.episode_main_category(ep)
            for ep in scenes.get(scene, {}).get("episodes", [])
        )
        quotas = {
            cat: max(0, min(n - existing[cat], self.num_eps_per_cat - counts[cat]))
            for n, cat, _ in self.load_ref_scene_eps(scene)
        }
        if not any(quotas.values()):
            print(f"No planned deficit for scene {scene}")
            return
        try:
            self._generate_and_save_scene(scene, scenes, quotas)
        finally:
            self._close_sim()

    def valid_categories(self):
        if not hasattr(self, "_valid_categories"):
            splits = self.open_json(self.cat_splits_path)["valid"]
            self._valid_categories = [
                cat for level in ("easy", "medium", "hard") for cat in splits[level]
            ]
        return self._valid_categories

    def episode_main_category(self, episode):
        category = episode["object_category"].split(",", 1)[0]
        if category not in self.valid_categories():
            raise ValueError(f"Unknown main category in saved episode: {category}")
        return category

    def read_saved_scenes(self):
        """Read scene outputs only; allocation and audit files live under logs/."""
        scenes = {}
        for filename in sorted(os.listdir(self.save_dir)):
            if not filename.endswith(".json.gz"):
                continue
            path = os.path.join(self.save_dir, filename)
            # Fail loudly on corrupt files; never silently replace existing data.
            with gzip.open(path, "rt") as f:
                data = json.load(f)
            if "episodes" in data and "goals_by_category" in data:
                scenes[filename[:-8]] = data
        return scenes

    def episode_counts(self, scenes):
        counts = Counter({cat: 0 for cat in self.valid_categories()})
        for data in scenes.values():
            for episode in data["episodes"]:
                counts[self.episode_main_category(episode)] += 1
        return counts

    def write_scene(self, scene, data):
        """Assign scene-local IDs and atomically replace the compressed output."""
        for index, episode in enumerate(data["episodes"]):
            episode["episode_id"] = index
        path = os.path.join(self.save_dir, f"{scene}.json.gz")
        temporary = path + ".tmp"
        with gzip.open(temporary, "wt") as f:
            json.dump(data, f)
        os.replace(temporary, path)

    def trim_excess_episodes(self, scenes):
        """Remove episodes from the most represented subcategories first."""
        counts = self.episode_counts(scenes)
        subcounts = Counter(
            ep["object_category"] for data in scenes.values() for ep in data["episodes"]
        )
        changed = set()
        for category, count in counts.items():
            for _ in range(max(0, count - self.num_eps_per_cat)):
                candidates = [
                    (scene, index, ep)
                    for scene, data in scenes.items()
                    for index, ep in enumerate(data["episodes"])
                    if self.episode_main_category(ep) == category
                ]
                scene, index, ep = max(
                    candidates, key=lambda item: subcounts[item[2]["object_category"]]
                )
                subcounts[ep["object_category"]] -= 1
                del scenes[scene]["episodes"][index]
                changed.add(scene)
        for scene in changed:
            self.write_scene(scene, scenes[scene])

    def save_objectnav_dataset(self, ref_scene_eps=None, ignore_scenes=None):
        """Resume from actual saved counts and redistribute only remaining deficits."""
        ignored = set(ignore_scenes or [])
        scenes = self.read_saved_scenes()
        self.trim_excess_episodes(scenes)
        self.subcategory_counts = Counter(
            ep["object_category"] for data in scenes.values() for ep in data["episodes"]
        )
        refs = self.open_json(self.cat_scene_inst_path)
        # Use the original eligibility inventory, not the failure-mutated log.
        eligible = self.open_json(self.cat_scene_inst_counts_path)
        plan = {}
        for cat in self.valid_categories():
            scene_ids = set(eligible.get(cat, {})) | {
                scene for scene in scenes if scene in refs.get(cat, {})
            }
            for scene in sorted(scene_ids - ignored):
                if scene in refs.get(cat, {}):
                    plan.setdefault(scene, []).append([0, cat, list(refs[cat][scene])])
        if ref_scene_eps is not None:
            # Reference counts are preferences only; retain fallback scenes.
            for scene, rows in plan.items():
                preferred = {
                    cat: count for count, cat, _ in ref_scene_eps.get(scene, [])
                }
                for row in rows:
                    row[0] = preferred.get(row[1], 0)
        self.save_json(
            os.path.join(self.save_dir, "logs", "scene_ep_counts.json"), plan
        )
        self.log_info = {"goals_by_category": {}, "episodes": {}}
        try:
            for round_index in range(self.generation_rounds):
                counts = self.episode_counts(scenes)
                if all(
                    counts[cat] == self.num_eps_per_cat
                    for cat in self.valid_categories()
                ):
                    break
                for scene, rows in plan.items():
                    counts = self.episode_counts(scenes)
                    existing = Counter(
                        self.episode_main_category(ep)
                        for ep in scenes.get(scene, {}).get("episodes", [])
                    )
                    quotas = {}
                    for preference, cat, _ in rows:
                        remaining = max(0, self.num_eps_per_cat - counts[cat])
                        # First pass respects the scene plan; later passes fill
                        # global deficits using any scene that can make progress.
                        quota = (
                            min(remaining, max(0, preference - existing[cat]))
                            if round_index == 0
                            else remaining
                        )
                        if quota:
                            quotas[cat] = quota
                    if not quotas:
                        continue
                    self._generate_and_save_scene(scene, scenes, quotas)
        finally:
            self._close_sim()
        self.audit_episode_counts()

    def audit_episode_counts(self):
        """Re-read saved files, write the count audit, and raise on any deficit."""
        counts = self.episode_counts(self.read_saved_scenes())
        deficits = {
            cat: self.num_eps_per_cat - counts[cat]
            for cat in self.valid_categories()
            if counts[cat] != self.num_eps_per_cat
        }
        self.save_json(
            os.path.join(self.save_dir, "logs", "episode_count_audit.json"),
            {
                "counts": dict(counts),
                "total": sum(counts.values()),
                "deficits": deficits,
                "complete": not deficits,
            },
        )
        if deficits:
            raise RuntimeError(
                f"Episode generation incomplete after bounded retries: {deficits}. See logs for failure reasons."
            )
        print(
            f"Verified {sum(counts.values())} saved episodes; {self.num_eps_per_cat} per category."
        )

    # Legacy geometry helpers remain available for external notebooks/scripts.
    # The episode pipeline above uses sample_start_for_goals instead.
    def is_point_valid(
        self, pt: np.ndarray, floor_check: bool = False, ref_pt: np.ndarray = None
    ):
        r"""
        Checks if a given (sampled) point is valid or not
        Args:
            - pt: Sampled point
            - floor_check: Check if there exists a path between pt and ref_pt, and whether their connecting path lies on the same floor
            - ref_pt: Reference point on the same floor as pt
        """

        if pt is None:
            return False

        if np.any(np.isnan(pt)):
            return False

        if floor_check:

            assert (
                ref_pt is not None
            ), "For enabling floor check, please provide a reference point on the floor."

            # Check for existing path between pt and ref_pt
            path = habitat_sim.ShortestPath()
            path.requested_start = ref_pt
            path.requested_end = pt
            found_path = self.sim.pathfinder.find_path(path)
            if not found_path:
                return False

            # Check if path points lie on the same floor
            heights = [path_pt.tolist()[1] for path_pt in path.points]
            height_diff = max(heights) - min(heights)
            if height_diff > self.floor_thresh:
                return False

        return True

    def get_radial_random_nav_pt(
        self,
        center_pt: np.ndarray,
        min_radius: float,
        soft_max_radius: float,
        hard_max_radius: float,
        num: int = 1,
        same_floor_check: bool = False,
        _recursive: bool = True,
    ):
        r"""
        Args:
            - center_pt: Sample points around this center point
            - min_radius: Minimum Radial distance of sampled points
            - soft_max_radius: Maximum Radial distance of sampled points
            - hard_max_radius: If soft_max_radius doesn't lead to enough samples, increase the max radius limit till this threshold
            - num: Number of samples
            - _recursive: If False, return one batch without expanding the radius.
        """

        snap_center_pt = self.sim.pathfinder.snap_point(center_pt)

        island_ind = self.sim.pathfinder.get_island(snap_center_pt)
        rand_pts = [
            self.sim.pathfinder.get_random_navigable_point_near(
                circle_center=snap_center_pt,  # Sample points around center_pt
                radius=soft_max_radius,  # Sample points within high_radius
                island_index=island_ind,
            )
            for _ in range(10 * num)
        ]  # Legacy helper: ten proposals per requested point

        snapped_pts = [
            self.sim.pathfinder.snap_point(pt)
            for pt in rand_pts  # Ignore None or NaN points, and snap the filtered points
            if self.is_point_valid(
                pt, floor_check=same_floor_check, ref_pt=snap_center_pt
            )
        ]

        # Points with distance more than <low_radius> from center point are considered valid
        dist_from_center = [
            self._geodesic_distance_from_view_pts(self.sim, snap_center_pt, pt)[1]
            for pt in snapped_pts
        ]
        valid_rand_pts = [
            snapped_pts[i]
            for i in range(len(snapped_pts))
            if dist_from_center[i] >= min_radius
        ]
        valid_rand_pts = valid_rand_pts[:num]

        # Sample new points and append them in if they are valid
        incr_max_radius = soft_max_radius
        incr_amount = (hard_max_radius - soft_max_radius) / self.num_incr_steps

        while _recursive and (len(valid_rand_pts) < num):

            incr_max_radius = (
                incr_max_radius + incr_amount
            )  # Increment the upper limit radius to sample more viewpoints

            # If hard_max_radius is surpassed, gathering all viewpoints is not successful, return None
            if incr_max_radius >= hard_max_radius:
                if len(valid_rand_pts) < num:
                    return None
                return valid_rand_pts

            extra_rand_pts = self.get_radial_random_nav_pt(
                snap_center_pt,
                min_radius,
                incr_max_radius,
                hard_max_radius,
                num - len(valid_rand_pts),
                _recursive=False,
            )

            # If newly generated pt not in valid_rand_pts, then add it in
            for pt in extra_rand_pts:
                if all(
                    [not np.array_equal(pt, valid_pt) for valid_pt in valid_rand_pts]
                ):
                    valid_rand_pts.append(pt)

        return valid_rand_pts

    @staticmethod
    def _geodesic_distance_from_view_pts(
        sim: Simulator,
        position_a: Union[Sequence[float], np.ndarray],
        view_pts: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
    ) -> Tuple[bool, float]:
        path = habitat_sim.MultiGoalShortestPath()
        if isinstance(view_pts[0], (Sequence, np.ndarray)):
            path.requested_ends = np.array(view_pts, dtype=np.float32)
        else:
            path.requested_ends = np.array([np.array(view_pts, dtype=np.float32)])
        path.requested_start = np.array(position_a, dtype=np.float32)

        found_path = sim.pathfinder.find_path(path)
        return found_path, path.geodesic_distance

    @staticmethod
    def face_object(object_position: np.ndarray, point: np.ndarray):
        EPS_ARRAY = np.array([1e-8, 0.0, 1e-8])
        cam_normal = (object_position - point) + EPS_ARRAY
        cam_normal[1] = 0
        cam_normal = cam_normal / np.linalg.norm(cam_normal)
        q = utils.quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)
        return utils.quat_to_coeffs(q)

    @staticmethod
    def dist_btw_pts(pt_0, pt_1):
        if not isinstance(pt_0, np.ndarray):
            pt_0 = np.array(pt_0)
        if not isinstance(pt_1, np.ndarray):
            pt_1 = np.array(pt_1)
        return np.linalg.norm(pt_1 - pt_0)

    @staticmethod
    def padded_int(num, pad=4):

        assert (
            isinstance(num, int) and num >= 0
        ), "Provided input must be a positive integer!"
        num_str = str(num)

        if pad <= len(num_str):
            return num_str
        return "0" * (pad - len(num_str)) + num_str

    @staticmethod
    def zip_and_save(d: Dict, save_path: str, overwrite: bool = False):

        assert (
            save_path.split(".")[-2] == "json" and save_path.split(".")[-1] == "gz"
        ), "Please provide a save_path as <file_name>.json.gz"

        if (
            not overwrite
            and os.path.exists(save_path)
            and os.path.getsize(save_path) > 0
        ):
            print(f"File already exists at path: {save_path}. Skipping...")
            return

        with gzip.open(save_path, "wt") as f:
            json.dump(d, f)
            print(f"Saved file to : {save_path}")

    @staticmethod
    def unzip_and_open(path: str):

        assert path.endswith("json.gz"), "Only json.gz files are supported."

        try:
            with gzip.open(path, "rt") as f:
                return json.load(f)
        except:
            return None

    @staticmethod
    def open_json(f_path):

        with open(f_path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json(f_path, info):
        assert f_path.endswith(".json")

        with open(f_path, "w") as f:
            json.dump(info, f, indent=4)


def main():
    """Parse CLI settings and run either a scene top-up or full reconciliation."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_eps_per_cat",
        type=int,
        default=50,
        help="Number of episodes to generate per category",
    )
    parser.add_argument("--scene_dataset_dir", type=str, default=SCENE_DATASET_ROOT_DIR)
    parser.add_argument("--task_dataset_dir", type=str, default=DATASET_ROOT_DIR)
    parser.add_argument(
        "--split", type=str, default="val", help="Dataset split to generate"
    )
    parser.add_argument(
        "--ignore_scenes",
        action="store_true",
        help="Ignores scenes mentioned in ignore_scenes.txt at dataset dir",
    )

    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Test Case Run to make sure pipeline works and output is as desired",
    )
    parser.add_argument(
        "--test_scene", type=str, default="102816036", help="Scene to use for test run"
    )
    parser.add_argument(
        "--save_dir", type=str, default=SAVE_DIR, help="Saves results here."
    )
    parser.add_argument("--agent_height", type=float, default=0.88)
    parser.add_argument("--agent_radius", type=float, default=0.18)
    parser.add_argument("--sensor_height", type=float, default=0.88)
    parser.add_argument(
        "--hfov", type=float, default=79.0, help="Horizontal Field of Vision"
    )
    parser.add_argument("--img_size", nargs=2, type=int, default=(480, 640))

    parser.add_argument(
        "--soft_max_radius",
        type=float,
        default=1.0,
        help="Legacy compatibility option; unused by this generator",
    )
    parser.add_argument(
        "--hard_max_radius",
        type=float,
        default=1.5,
        help="Legacy compatibility option; unused by this generator",
    )
    parser.add_argument(
        "--num_incr_steps",
        type=int,
        default=3,
        help="Legacy radial-helper setting; unused by episode sampling",
    )
    parser.add_argument(
        "--start_pos_min_radius",
        type=float,
        default=2.0,
        help="Minimum geodesic distance to the nearest goal viewpoint",
    )
    parser.add_argument(
        "--start_pos_max_radius",
        type=float,
        default=10.0,
        help="Maximum geodesic distance to the nearest goal viewpoint",
    )
    parser.add_argument(
        "--eps_per_obj",
        type=int,
        default=1,
        help="Legacy option; episode quotas now apply per main category",
    )
    parser.add_argument(
        "--floor_thresh",
        type=float,
        default=0.25,
        help="Floor Threshold beyond which two points are said to be in different floors.",
    )
    parser.add_argument(
        "--start_sample_attempts",
        type=int,
        default=100,
        help="Start proposals per sampling call",
    )
    parser.add_argument(
        "--candidate_retries",
        type=int,
        default=3,
        help="Failed sampling calls per subcategory per scene pass",
    )
    parser.add_argument(
        "--generation_rounds",
        type=int,
        default=3,
        help="Scene passes, including the initial allocation pass",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Legacy compatibility option; logging is always enabled",
    )

    args = parser.parse_args()

    # Create save directory and logs directory if they don't exist
    os.makedirs(os.path.join(args.save_dir, "logs"), exist_ok=True)

    generator = HSSD_Rare_Generator(
        num_eps_per_cat=args.num_eps_per_cat,
        agent_height=args.agent_height,
        agent_radius=args.agent_radius,
        sensor_height=args.sensor_height,
        hfov=args.hfov,
        img_size=args.img_size,
        scene_dataset_dir=args.scene_dataset_dir,
        dataset_root_dir=args.task_dataset_dir,
        save_dir=args.save_dir,
        data_split=args.split,
        num_incr_steps=args.num_incr_steps,
        start_pos_min_radius=args.start_pos_min_radius,
        start_pos_max_radius=args.start_pos_max_radius,
        eps_per_obj=args.eps_per_obj,
        floor_thresh=args.floor_thresh,
        verbose=args.verbose,
        start_sample_attempts=args.start_sample_attempts,
        candidate_retries=args.candidate_retries,
        generation_rounds=args.generation_rounds,
    )

    # Generate scene episode counts for reference in sampling episodes for each scene.
    ref_scene_eps = os.path.join(args.save_dir, "logs", "scene_ep_counts.json")
    if not os.path.exists(ref_scene_eps):
        generator.generate_scene_ep_counts()

    # Open ref_scene_eps
    with open(ref_scene_eps, "r") as f:
        ref_scene_eps = json.load(f)

    if args.test_run:

        print(f"Running Test Case...")
        generator.save_objectnav_dataset_for_scene(args.test_scene)

    else:
        ignore_scenes = []
        if args.ignore_scenes:
            ignore_scenes_path = os.path.join(
                generator.dataset_path, "ignore_scenes.txt"
            )
            with open(ignore_scenes_path, "r") as f:
                ignore_scenes = [name.strip() for name in f if name.strip()]
        print("Generating HSSD-Rare Dataset...")
        generator.save_objectnav_dataset(ref_scene_eps, ignore_scenes=ignore_scenes)


if __name__ == "__main__":
    main()
