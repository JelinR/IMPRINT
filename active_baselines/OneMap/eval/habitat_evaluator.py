# eval utils
from eval import get_closest_dist, FMMPlanner
from eval.actor import Actor
from eval.dataset_utils.gibson_dataset import load_gibson_episodes
from mapping import rerun_logger
from config import EvalConf
from onemap_utils import monochannel_to_inferno_rgb
from eval.dataset_utils import *

# os / filsystem
import bz2
import os
from os import listdir
import gzip
import json
import pathlib

# cv2
import cv2

# numpy
import numpy as np

# skimage
import skimage


# dataclasses
from dataclasses import dataclass

# quaternion
import quaternion

# typing
from typing import Tuple, List, Dict
import enum

# habitat
import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from habitat_sim.utils import common as utils

# tabulate
from tabulate import tabulate

# rerun
import rerun as rr

# pandas
import pandas as pd

# pickle
import pickle

# scipy
from scipy.spatial.transform import Rotation as R

from time import time

#TODO Changed: OmegaConf
from omegaconf import OmegaConf

#TODO Changed
# class TorchActionIDs:
#     STOP = torch.tensor([[0]], dtype=torch.long)
#     MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
#     TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
#     TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)

# class ActionIDs:
#     stop = 0
#     move_forward = 1
#     turn_left = 2
#     turn_right = 3

Action_ID_to_str = {
    0 : "stop",
    1 : "move_forward",
    2 : "turn_left",
    3 : "turn_right" 
}

class Result(enum.Enum):
    SUCCESS = 1
    FAILURE_MISDETECT = 2
    FAILURE_STUCK = 3
    FAILURE_OOT = 4
    FAILURE_NOT_REACHED = 5
    FAILURE_ALL_EXPLORED = 6
    FAILURE_ERROR = 7
    FAILURE_OBJ_ABSENT = 8

class HabitatEvaluator:
    def __init__(self,
                 config: EvalConf,
                 actor: Actor,
                 ) -> None:
        self.config = config
        self.multi_object = config.multi_object
        self.max_steps = config.max_steps
        self.max_dist = config.max_dist
        self.controller = config.controller
        self.mapping = config.mapping
        self.planner = config.planner
        self.log_rerun = config.log_rerun
        self.object_nav_path = config.object_nav_path
        self.scene_path = config.scene_path
        self.scene_data = {}
        self.episodes = []
        self.exclude_ids = []
        self.is_gibson = config.is_gibson

        #TODO Changed
        self.is_hssd = config.is_hssd
        self.is_trial = config.is_trial

        self.sim = None
        self.actor = actor
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True
        self.control_frequency = config.controller.control_freq
        self.max_vel = config.controller.max_vel
        self.max_ang_vel = config.controller.max_ang_vel
        self.time_step = 1.0 / self.control_frequency
        if self.is_gibson:
            dataset_info_file = str(pathlib.Path(self.object_nav_path).parent.absolute()) + \
                                "/val_info.pbz2".format(split="val")
            with bz2.BZ2File(dataset_info_file, 'rb') as f:
                self.dataset_info = pickle.load(f)
        else:
            self.dataset_info = None
        if self.is_gibson:
            self.episodes, self.scene_data = GibsonDataset.load_gibson_episodes(self.episodes,
                                                                                self.scene_data,
                                                                                self.dataset_info,
                                                                                self.object_nav_path)
        else:
            if self.multi_object:
                self.episodes, self.scene_data = HM3DMultiDataset.load_hm3d_multi_episodes(self.episodes,
                                                                                           self.scene_data,
                                                                                           self.object_nav_path)
            else:
                self.episodes, self.scene_data = HM3DDataset.load_hm3d_episodes(self.episodes,
                                                                                self.scene_data,
                                                                                self.object_nav_path)
        if self.actor is not None:
            self.logger = rerun_logger.RerunLogger(self.actor.mapper, False, "") if self.log_rerun else None
        # self.results_path = "/home/finn/active/MON/results_gibson" if self.is_gibson else "results/"

        #TODO Changed
        if self.is_gibson: self.results_path = "/mnt/OneMap/results/gibson"
        # elif self.is_hssd: self.results_path = f"/mnt/OneMap/results/hssd/{config.results_dir}"
        elif self.is_trial: self.results_path = "/mnt/OneMap/results/junk"
        elif "ovon" in self.object_nav_path: self.results_path = config.results_dir
        else:
            self.results_path = config.results_dir
            
        # self.results_path = "/mnt/OneMap/results/gibson" if self.is_gibson else "/mnt/OneMap/results/hm3d"
        
        self.use_saved_steps = config.use_saved_steps
        self.saved_steps_dir = config.saved_steps_dir
        self.saved_steps = None

        if self.use_saved_steps:
            assert os.path.exists(self.saved_steps_dir), f"Saved steps dir does not exist: {self.saved_steps_dir}"


        # if self.log_rerun:                                          #TODO Addd: REMOVE
        #     os.makedirs(self.results_path, exist_ok=True)
        #     rr.save(os.path.join(self.results_path, "rr_run.rrd"))

    def load_scene(self, scene_id: str):
        if self.sim is not None:
            self.sim.close()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        # backend_cfg.scene_id = self.scene_path + scene_id
        backend_cfg.scene_id = os.path.join(self.scene_path, scene_id)       #TODO Changed

        #TODO Changed: Adeed extra args
        ####
        # backend_cfg.allow_sliding = False
        # backend_cfg.pbr_image_based_lighting = True
        ####

        if self.is_gibson:
            pass # TODO
        elif self.is_hssd:
            hssd_dir = os.path.dirname(self.scene_path)
            backend_cfg.scene_dataset_config_file = os.path.join(hssd_dir, "hssd-hab.scene_dataset_config.json")
        else:
            backend_cfg.scene_dataset_config_file = self.scene_path + "hm3d/hm3d_annotated_basis.scene_dataset_config.json" #TODO: Scene Dataset Config is loaded here


        #Config for Saved Navigation
        if (self.use_saved_steps):
            
            #Config for Saved Navigation in HM3D (Multi-Floor)
            if not (self.is_hssd) and not (self.is_gibson):
                hfov = 90
                rgb = habitat_sim.CameraSensorSpec()
                rgb.uuid = "rgb"
                rgb.hfov = hfov
                rgb.position = np.array([0, 1.1, 0])                    #Changed: 0.88 -> 1.1
                rgb.sensor_type = habitat_sim.SensorType.COLOR
                res = 640
                rgb.resolution = [res, res]

                depth = habitat_sim.CameraSensorSpec()
                depth.uuid = "depth"
                depth.hfov = hfov
                depth.sensor_type = habitat_sim.SensorType.DEPTH
                depth.position = np.array([0, 1.1, 0])                    #Changed: 0.88 -> 1.1
                depth.resolution = [res, res]

                agent_cfg = habitat_sim.agent.AgentConfiguration(
                    height = 1.1,
                    radius = 0.08,
                    action_space=dict(
                        move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
                        turn_left=ActionSpec("turn_left", ActuationSpec(amount=20.0)),              #Changed: 5 -> 20
                        turn_right=ActionSpec("turn_right", ActuationSpec(amount=20.0)),            #Changed: 5 -> 20
                    )
                )

            #Config for Saved Navigation in HSSD (Overmapped)
            elif self.is_hssd:
                hfov = 90
                rgb = habitat_sim.CameraSensorSpec()
                rgb.uuid = "rgb"
                rgb.hfov = hfov
                rgb.position = np.array([0, 0.88, 0])
                rgb.sensor_type = habitat_sim.SensorType.COLOR
                res = 640
                rgb.resolution = [res, res]

                depth = habitat_sim.CameraSensorSpec()
                depth.uuid = "depth"
                depth.hfov = hfov
                depth.sensor_type = habitat_sim.SensorType.DEPTH
                depth.position = np.array([0, 0.88, 0])
                depth.resolution = [res, res]

                agent_cfg = habitat_sim.agent.AgentConfiguration(
                    height = 1.1,
                    radius = 0.08,
                    action_space=dict(
                        move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
                        turn_left=ActionSpec("turn_left", ActuationSpec(amount=30.0)),              #Changed: 5 -> 30
                        turn_right=ActionSpec("turn_right", ActuationSpec(amount=30.0)),            #Changed: 5 -> 30
                    )
                )


            else:
                pass

        
        else:
            #OG Config
            hfov = 90
            rgb = habitat_sim.CameraSensorSpec()
            rgb.uuid = "rgb"
            rgb.hfov = hfov
            rgb.position = np.array([0, 0.88, 0])
            rgb.sensor_type = habitat_sim.SensorType.COLOR
            res = 640
            rgb.resolution = [res, res]

            depth = habitat_sim.CameraSensorSpec()
            depth.uuid = "depth"
            depth.hfov = hfov
            depth.sensor_type = habitat_sim.SensorType.DEPTH
            depth.position = np.array([0, 0.88, 0])
            depth.resolution = [res, res]

            agent_cfg = habitat_sim.agent.AgentConfiguration(action_space=dict(
                move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
                turn_left=ActionSpec("turn_left", ActuationSpec(amount=5.0)),
                turn_right=ActionSpec("turn_right", ActuationSpec(amount=5.0)),
            ))

    
        agent_cfg.sensor_specifications = [rgb, depth]
        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        print_hab_cfg(sim_cfg)
        
        self.sim = habitat_sim.Simulator(sim_cfg)                           #TODO: Sim is Loaded here

        #TODO Changed: Added navmesh settings
        ####
        # navmesh_settings = habitat_sim.NavMeshSettings()
        # navmesh_settings.set_defaults()
        # navmesh_settings.agent_height = 0.88
        # navmesh_settings.agent_radius = 0.18
        # # navmesh_settings.agent_max_climb = 0.2
        # # navmesh_settings.cell_height = 0.2
        # navmesh_settings.include_static_objects = True
        # navmesh_success = self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
        # print(f"Navmesh Recomputed: {navmesh_success}")
        ####

        # print_hab_cfg(self.sim.config)


        if self.scene_data[scene_id].objects_loaded:
            return
        if not self.is_gibson:
            self.scene_data = HM3DDataset.load_hm3d_objects(self.scene_data, self.sim.semantic_scene.objects, scene_id)
        else:
            self.scene_data = GibsonDataset.load_gibson_objects(self.scene_data, self.dataset_info, scene_id)

    def execute_action(self, action: Dict
                       ):
        if 'discrete' in action.keys():
            # We have a discrete actor
            self.sim.step(action['discrete'])

        elif 'continuous' in action.keys():
            # We have a continuous actor
            self.vel_control.angular_velocity = action['continuous']['angular']
            self.vel_control.linear_velocity = action['continuous']['linear']
            agent_state = self.sim.get_agent(0).state
            previous_rigid_state = habitat_sim.RigidState(
                utils.quat_to_magnum(agent_state.rotation), agent_state.position
            )

            # manually integrate the rigid state
            target_rigid_state = self.vel_control.integrate_transform(
                self.time_step, previous_rigid_state
            )

            # snap rigid state to navmesh and set state to object/sim
            # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
            end_pos = self.sim.step_filter(
                previous_rigid_state.translation, target_rigid_state.translation
            )

            # set the computed state
            agent_state.position = end_pos
            agent_state.rotation = utils.quat_from_magnum(
                target_rigid_state.rotation
            )
            self.sim.get_agent(0).set_state(agent_state)
            self.sim.step_physics(self.time_step)

    def read_results(self, path, sort_by):
        state_dir = os.path.join(path, 'state')
        state_results = {}
        object_query = {}
        scene_name = {}
        spl = {}

        # Check if the state directory exists
        if not os.path.isdir(state_dir):
            print(f"Error: {state_dir} is not a valid directory")
            return state_results
        pose_dir = os.path.join(os.path.abspath(os.path.join(state_dir, os.pardir)), "trajectories")

        # Iterate through all files in the state directory
        for filename in os.listdir(state_dir):
            if filename.startswith('state_') and filename.endswith('.txt'):
                try:
                    # Extract the experiment number from the filename
                    experiment_num = int(filename[6:-4])  # removes 'state_' and '.txt'
                    # if experiment_num > 1045:
                    #     continue
                    # Read the content of the file
                    with open(os.path.join(state_dir, filename), 'r') as file:
                        content = file.read().strip()

                    # Convert the content to a number (assuming it's a float)
                    state_value = int(content)
                    # Store the result in the dictionary
                    state_results[experiment_num] = state_value
                    object_query[experiment_num] = self.episodes[experiment_num].obj_sequence[0]
                    scene_name[experiment_num] = self.episodes[experiment_num].scene_id
                    poses = np.genfromtxt(os.path.join(pose_dir, "poses_" + str(experiment_num) + ".csv"), delimiter=",")
                    deltas = poses[1:, :2] - poses[:-1, :2]
                    distance_traveled = np.linalg.norm(deltas, axis=1).sum()
                    if state_value == 1:
                        spl[experiment_num] = self.episodes[experiment_num].best_dist / max(self.episodes[experiment_num].best_dist, distance_traveled)
                    else:
                        spl[experiment_num] = 0
                    if self.episodes[experiment_num].episode_id != experiment_num:
                        print(f"Warning, exerpiment_num {experiment_num} does not correctly resolve to episode_id {self.episodes[experiment_num].episode_id}")
                except ValueError:
                    print(f"Warning: Skipping {filename} due to invalid format")
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
        dict_res = {"state": state_results, "obj" : object_query, "scene" : scene_name, "spl" : spl}
        data = pd.DataFrame.from_dict(dict_res)

        states = data["state"].unique()

        def calculate_percentages(group):
            total = len(group)
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / total for state in states})

            # Calculate average SPL and multiply by 100
            avg_spl = group['spl'].mean()
            result['Average SPL'] = avg_spl

            return result

        # Per-object results
        object_results = data.groupby('obj').apply(calculate_percentages).reset_index()
        object_results = object_results.rename(columns={'obj': 'Object'})

        # Per-scene results
        scene_results = data.groupby('scene').apply(calculate_percentages).reset_index()
        scene_results = scene_results.rename(columns={'scene': 'Scene'})

        # Overall results
        overall_percentages = calculate_percentages(data)
        overall_row = pd.DataFrame([{'Object': 'Overall'} | overall_percentages.to_dict()])
        object_results = pd.concat([overall_row, object_results], ignore_index=True)

        overall_row = pd.DataFrame([{'Scene': 'Overall'} | overall_percentages.to_dict()])
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        # Sorting
        object_results = object_results.sort_values(by=sort_by, ascending=False)
        scene_results = scene_results.sort_values(by=sort_by, ascending=False)

        # Function to format percentages
        def format_percentages(val):
            return f"{val:.2%}" if isinstance(val, float) else val

        # Apply formatting to all columns except the first one (Object/Scene)
        object_table = object_results.iloc[:, 0].to_frame().join(
            object_results.iloc[:, 1:].applymap(format_percentages))
        scene_table = scene_results.iloc[:, 0].to_frame().join(
            scene_results.iloc[:, 1:].applymap(format_percentages))

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))
        return data

    def evaluate(self):
        success = 0
        n_eps = 0
        # randomly shuffle episodes
        # random.shuffle(self.episodes)
        success_per_obj = {}
        obj_count = {}
        results = []

        #TODO Added: Track scenes, episodes already run
        ####
        results_dir = os.path.join(self.results_path, "state")
        os.makedirs(results_dir, exist_ok=True)
        

        #get_scene_eps = lambda f: (f.split("_")[1], int(f.split("_")[2].split(".")[0]))
        # get_done_eps = lambda f: ("_".join( f.split("state_")[-1].split("_")[:-1] ), int( f.split("_")[-1].split(".txt")[0] ))
        get_done_eps = lambda f: int( f.split("_")[-1].split(".txt")[0] )

        scene_eps_done = [get_done_eps(f) for f in os.listdir(results_dir)]

        # ref_results_dir = os.path.join("results/hssd_onemap_temp", "state")
        # ref_poses_dir = os.path.join("results/hssd_onemap_temp", "trajectories")
        # scene_eps_done = []
        # scene_eps_all = []
        # for f in os.listdir(ref_results_dir):

        #     scene_eps = get_scene_eps(f)
        #     scene_eps_all.append(scene_eps)

        #     pose_file = f"poses_{scene_eps[1]}.csv"
        #     pose_file_path = os.path.join(ref_poses_dir, pose_file)
        #     if os.path.exists(pose_file_path): scene_eps_done.append(scene_eps)


        print(f"Finished (Scene, Episode)s: {len(scene_eps_done)}/{len(self.episodes)}\n{scene_eps_done}")


        #Update results with saved result values
        for ep_num in scene_eps_done:

            # saved_episode = [elem for elem in scene_eps_done if elem[1] == ep_num][0]
            state_path = os.path.join(results_dir, f"state_{ep_num}.txt")
            with open(state_path, "r") as f:
                state_result = f.readlines()

            results.append(int(state_result[0]))

        # for ep_num in range(len(os.listdir(ref_results_dir))):

        #     saved_scene, saved_episode = [elem for elem in scene_eps_all if elem[1] == ep_num][0]
        #     state_path = os.path.join(ref_results_dir, f"state_{saved_scene}_{saved_episode}.txt")
        #     with open(state_path, "r") as f:
        #         state_result = f.readlines()

        #     results.append(int(state_result[0]))

        print(f"Loaded Saved Results: {results}")
        ####

        step_times = []
        episode_times = []
        step_time_path = f"{self.results_path}/timings/step_times.txt"
        episode_time_path = f"{self.results_path}/timings/episode_times.txt"
        os.makedirs(os.path.dirname(step_time_path), exist_ok=True)

        #TODO ADDED: Train in multiple terminals for different episodes
        assert self.config.run_split in [0, 1, 2]

        if self.config.run_split == 1:
            self.exclude_ids = np.arange(300, len(self.episodes))
            print(f"Loading split 1 episodes...")
            assert "split_1" in self.config.results_dir

        elif self.config.run_split == 2:
            self.exclude_ids = np.arange(0, 300)
            print(f"Loading split 2 episodes...")
            assert "split_2" in self.config.results_dir

        
        # restart at 930
        for n_ep, episode in enumerate(self.episodes):
        # for n_ep, episode in enumerate(self.episodes[492:]):

            ###TODO Changed: Skip episodes already run
            # curr_eps = (episode.scene_id, episode.episode_id)
            # print(f"Current (Scene, Episode): {curr_eps}")

            # if curr_eps != ("102344094", 8):
            #     print(f'{curr_eps} is not ("102344094", 8). Skipping...')
            #     continue

            #Only evaluate specific episode
            # if n_ep != 0:
            #     continue

            if episode.episode_id in scene_eps_done:
                print(f"{episode.episode_id} already evaluated. Skipping...\n")
                continue

            #If HSSD or HM3D object not in saved scraped images dir, then continue
            # obj_category = episode.obj_sequence[0]
            # saved_hssd_objects = os.listdir("/mnt/vlfm_query_embed/data/scraped_imgs/hssd_15")
            # if (obj_category not in saved_hssd_objects) or (n_ep in [56, 57, 66, 74, 92, 100, 122, 133, 135, 139]): 
            #     print("Object not in Saved Data Directory! Skipping...")

            #     results.append(Result.FAILURE_OBJ_ABSENT)

            #     results_state_dir = os.path.join(self.results_path, "state")
            #     os.makedirs(results_state_dir, exist_ok=True) 

            #     with open(f"{results_state_dir}/state_{episode.scene_id}_{episode.episode_id}.txt", 'w') as f:
            #         f.write(str(results[n_ep].value))

            #     continue

            ###


            #####
            #TODO Changed: Make sure that saved steps file exists for this scene and starting pose
            if self.use_saved_steps:
                scene_path = episode.scene_id
                scene_id = scene_path.split("/")[-1].split(".")[0]
                episode_id = episode.episode_id
                start_pos, start_rot = episode.start_position, episode.start_rotation

                print(f"\nNew Scene or Episode!\nScene ID: {scene_id}, Episode ID: {episode_id}, start_pos: {start_pos}, start_rot: {start_rot}")

                scene_steps_files = [file for file in os.listdir(self.saved_steps_dir) if file.__contains__(scene_id)]
                print(f"Scene Step Files: {len(scene_steps_files)}")
                if len(scene_steps_files) == 0: 
                    print(f"No Files Found! Skipping...")
                    continue

                for file in scene_steps_files:

                    file_path = os.path.join(self.saved_steps_dir, file)
                    step_actions = dict(np.load(file_path, allow_pickle=True))
                    print(f"Saved Start Pos: {step_actions['init_pos_abs']}, Saved Start Rot: {step_actions['init_rot_abs']}")
                    if (step_actions["init_pos_abs"] == start_pos).all() and (step_actions["init_rot_abs"] == start_rot).all():

                        del step_actions['init_pos_abs']
                        del step_actions['init_rot_abs']
                        del step_actions['scene']
                        del step_actions['episode']

                        # step_actions = {int(k):torch.Tensor(v).to(dtype=torch.int64) for (k, v) in step_actions.items()}
                        # step_actions = {int(k): int(v[0][0]) for (k, v) in step_actions.items()}
                        step_actions = {int(k): Action_ID_to_str[ v[0][0] ] for (k, v) in step_actions.items()}

                        print(f"Found Saved Steps for Scene : {scene_id}")
                        skip_episode = False
                        break
                    else:
                        skip_episode = True

                if skip_episode: 
                    print(f"Saved Step Actions not found for Scene: {scene_id}, Episode: {episode_id}. Skipping... ")
                    continue

            #####

            start_episode_time = time()

            poses = []
            results.append(Result.FAILURE_OOT)      #TODO COMMENTED: Need to uncomment
            # results[n_ep] = Result.FAILURE_OOT
            steps = 0
            if n_ep in self.exclude_ids:
                continue
            n_eps += 1
            print(f"\n\nEpisode Num: {n_ep}")
            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)
            # if self.is_gibson:
            #     episode = self.compute_gt_path_gibson(episode)
            self.sim.initialize_agent(0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
            self.actor.reset()
            current_obj_id = 0
            current_obj = episode.obj_sequence[current_obj_id]
            if current_obj not in success_per_obj:
                success_per_obj[current_obj] = 0
                obj_count[current_obj] = 1
            else:
                obj_count[current_obj] += 1
            self.actor.set_query(current_obj)

            if self.log_rerun:

                pts = []
                for obj in self.scene_data[episode.scene_id].object_locations[current_obj]:
                    if not self.is_gibson:
                        pt = obj.bbox.center[[0, 2]]
                        pt = (-pt[1], -pt[0])
                        pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                    else:
                        for pt_ in obj:
                            pt = (pt_[0], pt_[1])
                            pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                pts = np.array(pts)
                rr.log("map/ground_truth", rr.Points2D(pts, colors=[[255, 255, 0]], radii=[1]))

            while steps < self.max_steps and current_obj_id < len(episode.obj_sequence):

                start_step_time = time()

                observations = self.sim.get_sensor_observations()
                # observations['depth'] = fill_depth_holes(observations['depth'])
                observations['state'] = self.sim.get_agent(0).get_state()
                pose = np.zeros((4, ))
                pose[0] = -observations['state'].position[2]
                pose[1] = -observations['state'].position[0]
                pose[2] = observations['state'].position[1]
                # yaw
                orientation = observations['state'].rotation
                q0 = orientation.x
                q1 = orientation.y
                q2 = orientation.z
                q3 = orientation.w
                r = R.from_quat([q0, q1, q2, q3])
                # r to euler
                yaw, _, _1 = r.as_euler("yxz")
                pose[3] = yaw

                poses.append(pose)
                if self.log_rerun:
                    cam_x = -self.sim.get_agent(0).get_state().position[2]
                    cam_y = -self.sim.get_agent(0).get_state().position[0]
                    rr.log("camera/rgb", rr.Image(observations["rgb"]).compress(jpeg_quality=50))
                    rr.log("camera/depth", rr.Image((observations["depth"] - observations["depth"].min()) / (
                            observations["depth"].max() - observations["depth"].min())))
                    self.logger.log_pos(cam_x, cam_y)
                    
                action, called_found = self.actor.act(observations)     #TODO: Actor Action takes place here

                #TODO Changed: Action from saved steps actions
                ####
                if self.use_saved_steps:
                    action = {}
                    saved_action = step_actions[steps]
                    if saved_action == "stop": 
                        break
                    else: 
                        action["discrete"] = saved_action
                ####
                
                self.execute_action(action)
                if self.log_rerun:
                    self.logger.log_map()
                
                #TODO: Added
                print(f"Step : {steps}, Action : {action}, Called Found: {called_found}")

                if called_found:
                    # We will now compute the closest distance to the bounding box of the object
                    #TODO: Gets Distance to Goal Object
                    # dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                    #                         self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)
                    
                    dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                            self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)
                    
                    if dist < self.max_dist:
                        results[n_ep] = Result.SUCCESS
                        success += 1
                        print("Object found!")
                        success_per_obj[current_obj] += 1
                    else:
                        pos = self.actor.mapper.chosen_detection
                        pos_metric = self.actor.mapper.one_map.px_to_metric(pos[0], pos[1])

                        #TODO COMMENTED
                        # dist_detect = get_closest_dist([-pos_metric[1], -pos_metric[0]],
                        #                     self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)
                        
                        dist_detect = get_closest_dist(
                            [-pos_metric[1], -pos_metric[0]],
                            self.scene_data[episode.scene_id].object_locations[current_obj],
                            self.is_gibson,
                        )

                        if dist_detect < self.max_dist:
                            results[n_ep] = Result.FAILURE_NOT_REACHED
                        else:
                            results[n_ep] = Result.FAILURE_MISDETECT
                        print(f"Object not found! Dist {dist}, detect dist: {dist_detect}.")
                    
                    current_obj_id += 1
                    if current_obj_id < len(episode.obj_sequence):
                        current_obj = episode.obj_sequence[current_obj_id]
                        if current_obj not in success_per_obj:
                            success_per_obj[current_obj] = 0
                            obj_count[current_obj] = 1
                            obj_count[current_obj] += 1
                        self.actor.set_query(current_obj)

                if steps % 100 == 0:
                    #TODO: Calculates Distance to Goal Object                        
                    # dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                    #                         self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)
                    
                    dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                            self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)


                    print(f"Step {steps}, current object: {current_obj}, episode_id: {episode.episode_id}, distance to closest object: {dist}")
                steps += 1

                finish_step_time = time()
                step_time = finish_step_time - start_step_time
                with open(step_time_path, "a") as f:
                    f.write(f"{step_time}\n")



            poses = np.array(poses)
            # If the last 10 poses didn't change much and we have OOT, assume stuck
            #TODO Changed: Commented 
            #########
            if results[n_ep] == Result.FAILURE_OOT and np.linalg.norm(poses[-1] - poses[-10]) < 0.05:
                results[n_ep] = Result.FAILURE_STUCK

            num_frontiers = len(self.actor.mapper.nav_goals)
            save_path = f"{self.results_path}/trajectories/poses_{episode.episode_id}.csv"  #TODO OG Changed: Added save_path
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            np.savetxt(save_path, poses, delimiter=",")
            print(f"\nSaved poses to {save_path}")
            #########

            # save final sim to image file
            final_sim = (self.actor.mapper.get_map() + 1.0) / 2.0
            final_sim = final_sim[0]
            final_sim = final_sim.transpose((1, 0))
            final_sim = np.flip(final_sim, axis=0)
            final_sim = monochannel_to_inferno_rgb(final_sim)
            save_path = f"{self.results_path}/similarities/final_sim_{episode.episode_id}.png"
            # save_path = f"{self.results_path}/similarities/final_sim_{episode.scene_id}_{episode.episode_id}.png"          #TODO Changed: Included scene info
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            
            cv2.imwrite(save_path, final_sim)
            print(f"Saved Image to {save_path}")


            #TODO Changed: Save Feature Map
            ####
            if self.use_saved_steps:
                save_embed_path = f"{self.results_path}/feature_maps/feature_map_scene_{scene_id}.npz"
                os.makedirs(os.path.dirname(save_embed_path), exist_ok = True)

                if os.path.exists(save_embed_path):
                    print(f"Saved Feature Map found at : {save_embed_path}")
                else:
                    np.savez(save_embed_path, arr=self.actor.mapper.one_map.feature_map)
                    print(f"Saved Feature Map to : {save_embed_path}")
            ####


            # break

            #TODO Changed: Commented 
            #########
            if (results[n_ep] == Result.FAILURE_STUCK or results[n_ep] == Result.FAILURE_OOT) and num_frontiers == 0:
                results[n_ep] = Result.FAILURE_ALL_EXPLORED
            print(f"Overall success: {success / (n_eps)}, per object: ")
            for obj in success_per_obj.keys():
                print(f"{obj}: {success_per_obj[obj] / obj_count[obj]}")
            print(
                f"Result distribution: successes: {results.count(Result.SUCCESS)}, misdetects: {results.count(Result.FAILURE_MISDETECT)}, OOT: {results.count(Result.FAILURE_OOT)}, stuck: {results.count(Result.FAILURE_STUCK)}, not reached: {results.count(Result.FAILURE_NOT_REACHED)}, all explored: {results.count(Result.FAILURE_ALL_EXPLORED)}")
            
            # Write result to file
            results_state_dir = os.path.join(self.results_path, "state")
            os.makedirs(results_state_dir, exist_ok=True) 

            with open(f"{results_state_dir}/state_{episode.episode_id}.txt", 'w') as f:
                f.write(str(results[n_ep].value))
            ###########

            finish_episode_time = time()
            episode_time = finish_episode_time - start_episode_time
            with open(episode_time_path, "a") as f:
                f.write(f"{episode_time}\n")

            # except Exception as e:
            #     print(f"Exception: {e}")

            #     #Save scene, episode where error occured
            #     exception_file_path = os.path.join(self.results_path, "errors.txt")
            #     with open(exception_file_path, "w") as f:
            #         f.write(f"{episode.scene_id}, {episode.episode_id}")

            #     #Update results with failure type as error
            #     results[n_ep] = Result.FAILURE_ERROR

            #     results_state_dir = os.path.join(self.results_path, "state")
            #     os.makedirs(results_state_dir, exist_ok=True) 

            #     with open(f"{results_state_dir}/state_{episode.scene_id}_{episode.episode_id}.txt", 'w') as f:
            #         f.write(str(results[n_ep].value))

            #     continue




#TODO Changed: Utils for printing config
print_yaml = lambda cfg: print(OmegaConf.to_yaml(cfg))

def get_object_attrs(obj):
    return [attr for attr in dir(obj) if not ((attr.startswith("__")) and (attr.endswith("__")))]


def print_hab_cfg(hab_cfg):

    cfg = {}

    sim_cfg = hab_cfg.sim_cfg
    agents_cfg = hab_cfg.agents

    sim_attrs = get_object_attrs(sim_cfg)
    cfg["sim_cfg"] = {}
    for attr in sim_attrs:
        cfg["sim_cfg"][attr] = sim_cfg.__getattribute__(attr)
        
    # print_yaml(cfg)
    num_agents = len(agents_cfg)
    for count in range(num_agents):
        cfg[f"agent_{count}"] = {}

        agent_cfg = agents_cfg[count]
        agent_attrs = get_object_attrs(agent_cfg)
        for attr in agent_attrs:

            if attr == "sensor_specifications":
                cfg[f"agent_{count}"]["sensors"] = {}
                for sensor_cfg in agent_cfg.__getattribute__(attr):
                    sensor_attrs = get_object_attrs(sensor_cfg)
                    sensor_uuid = sensor_cfg.uuid
                    cfg[f"agent_{count}"]["sensors"][sensor_uuid] = {}

                    for sensor_attr in sensor_attrs:
                        cfg[f"agent_{count}"]["sensors"][sensor_uuid][sensor_attr] = str(sensor_cfg.__getattribute__(sensor_attr))

            else:
                cfg[f"agent_{count}"][attr] = agent_cfg.__getattribute__(attr)

    
    print_yaml(cfg)

    return cfg