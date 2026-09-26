import math
import os
import random
from tqdm import tqdm

import magnum as mn
import numpy as np
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
from typing import Dict

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations import maps
from habitat_sim.utils import common as utils

import gzip
import json

from argparse import ArgumentParser


from typing import List


EPS_DIR = "./dataset/check"
SAVE_DIR = "./dataset/check_plot_vw_pts"
SCENE_DATASET_ROOT_DIR = "./habitat-lab/data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"

# --- Utils : General ----

def parse_args():

    parser = ArgumentParser()
    parser.add_argument("--episodes_dir", type=str, default=EPS_DIR)
    parser.add_argument("--scene_dataset_path", type=str, default=SCENE_DATASET_ROOT_DIR)
    parser.add_argument("--save_plots_dir", type=str, default=SAVE_DIR)
    args = parser.parse_args()

    return args

def open_zipped_json(f_path):

    with gzip.open(f_path, "r") as f:
        return json.load(f)

def make_sim(scene_id: str, scene_dataset_path: str, 
                override_sim_settings: Dict = {}):

    sim_settings = {
        "scene": scene_id,                        
        "scene_dataset": scene_dataset_path,             
        "default_agent": 0,                     # Index of the default agent
        "sensor_vert_position": 0.88,                  # Height of Sensor
        "agent_radius": 0.18,                   # Radius of Agent
        "width": 640,                           # Sensor Width
        "height": 480,                          # Sensor Height
        "hfov": 79,
        "agent_max_climb": 0.2,
        "cell_height": 0.2,
        "enable_physics": False,
        "gpu_device_id": 0
    }

    if len(override_sim_settings) > 0:
        for key in override_sim_settings:
            sim_settings[key] = override_sim_settings[key]
    
    #Build Simulator Configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = sim_settings["gpu_device_id"]
    sim_cfg.scene_id = sim_settings["scene"]
    sim_cfg.scene_dataset_config_file = sim_settings["scene_dataset"]
    sim_cfg.enable_physics = sim_settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [sim_settings["height"], sim_settings["width"]]
    color_sensor_spec.position = [0.0, sim_settings["sensor_vert_position"], 0.0]
    color_sensor_spec.hfov = sim_settings["hfov"]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [sim_settings["height"], sim_settings["width"]]
    depth_sensor_spec.position = [0.0, sim_settings["sensor_vert_position"], 0.0]
    depth_sensor_spec.hfov = sim_settings["hfov"]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    # semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    # semantic_sensor_spec.uuid = "semantic_sensor"
    # semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    # semantic_sensor_spec.resolution = [sim_settings["height"], sim_settings["width"]]
    # semantic_sensor_spec.position = [0.0, sim_settings["sensor_vert_position"], 0.0]
    # semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    # sensor_specs.append(semantic_sensor_spec)

    #Build Agent Configuration: Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    #Build Navmesh
    print(f"Pathfinder is Loaded: {sim.pathfinder.is_loaded}")
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_height = sim_settings["sensor_vert_position"]
    navmesh_settings.agent_radius = sim_settings["agent_radius"]
    navmesh_settings.agent_max_climb = sim_settings["agent_max_climb"]
    navmesh_settings.cell_height = sim_settings["cell_height"]
    navmesh_settings.include_static_objects = True
    navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    print(f"Navmesh Recompute Success: {navmesh_success}")

    return sim


# ----- Utils: Plotting -----


def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

def display_map(topdown_map, key_points=None, with_line=False, save_path=None):
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")

    plt.imshow(topdown_map)

    # plot points on map
    start_col = "#e07a5f"
    end_col = "#81b29a"
    mid_col = "#f2cc8f" #"yellow"
    if key_points is not None:
        for count, point_set in enumerate(key_points):
            if count == 0: col = start_col
            elif count == 1: col = mid_col
            else: col = end_col

            for point in point_set:

                plt.plot(point[0], point[1], marker="o", markersize=5, alpha=0.8, color = col)
                if count>0 and with_line:
                    plt.plot([key_points[count][0], key_points[count-1][0]],
                            [key_points[count][1], key_points[count-1][1]],
                            linestyle="dashed",
                            linewidth=2,
                            color="#f4f1de",
                            alpha = 0.5)

    if save_path is None:
        plt.show(block = False)

    if save_path is not None:
        plt.savefig(save_path)

        plt.close()
            
def plot_topdown_with_pts(sim, obj_pts, view_pts, alt_pts=None,
                        meters_per_pixel=0.05, 
                        with_line=False, snap_points=False,
                        save_path=None):

    vis_obj_points = convert_points_to_topdown(
        sim.pathfinder, 
        # snapped_pts if snap_points else pts, 
        obj_pts,
        meters_per_pixel
    )

    vis_view_points = convert_points_to_topdown(
        sim.pathfinder, 
        # snapped_pts if snap_points else pts, 
        view_pts,
        meters_per_pixel
    )

    if alt_pts is not None:
        vis_alt_points = convert_points_to_topdown(
            sim.pathfinder, 
            # snapped_pts if snap_points else pts, 
            alt_pts,
            meters_per_pixel
        )
    else:
        vis_alt_points = []

    vis_points = [vis_obj_points, vis_view_points, vis_alt_points]

    top_down_map = maps.get_topdown_map(
        sim.pathfinder, 
        height= view_pts[0][1],
        meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]

    display_map(top_down_map, key_points=vis_points, with_line=with_line, save_path=save_path)


# ---- Main ------

def main():

    args = parse_args()

    for scene_path in os.listdir(args.episodes_dir):

        if not scene_path.endswith(".json.gz"):
            continue

        scene_info = open_zipped_json(os.path.join(args.episodes_dir, scene_path))
        
        scene_name = scene_path.split(".")[0]
        print(f"\nPlotting for scene: {scene_name}...")

        #Create sim
        sim = make_sim(scene_id = scene_name,
                        scene_dataset_path = args.scene_dataset_path)

        #Loop through categories in scene
        for k in tqdm(scene_info["goals_by_category"].keys()):
            
            #Extract main category and category
            cat = k.replace(scene_name + "_", "")
            main_cat = cat.split(",")[0]

            #Create save plot dir
            save_dir = os.path.join(args.save_plots_dir, main_cat, scene_name, cat)
            os.makedirs(save_dir, exist_ok = True)

            #Get instance position and viewpoints
            cat_info = scene_info["goals_by_category"][k]
            inst_pos = []
            view_pts = []

            for cat_inst in cat_info:

                inst_pos.append( cat_inst["position"] )

                inst_vw_pts = [vw["agent_state"]["position"] for vw in cat_inst["view_points"]]
                view_pts.extend(inst_vw_pts)

            #Plotting
            save_path = os.path.join(save_dir, "inst_viewpoints.png")
            plot_topdown_with_pts(sim, 
                                obj_pts = inst_pos, 
                                view_pts = view_pts, 
                                save_path = save_path)

        sim.close()

    print(f"\n\nDone!")


if __name__ == "__main__":
    main()