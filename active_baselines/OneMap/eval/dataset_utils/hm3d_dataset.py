from eval.dataset_utils import Episode, SceneData, SemanticObject

# typing
from typing import Dict, List

# filesystem utils
import os
from os import listdir
import gzip
import json

def load_hm3d_episodes(episodes: List[Episode], scene_data: Dict[str, SceneData], object_nav_path: str):
    i = 0
    files = listdir(object_nav_path)

    ########
    #TODO: Changed

    #Ignore Scenes that are tough to explore
    ignore_scenes_path = os.path.join(os.path.dirname(object_nav_path[:-1]), "ignore_scenes.txt")
    if os.path.exists(ignore_scenes_path):
        with open(ignore_scenes_path, "r") as f:
            ignore_scenes = f.readlines()

        ignore_scenes = [name.strip() + ".json.gz" for name in ignore_scenes]
    else: 
        ignore_scenes = []
    scenes_count = 0

    #Track Scenes and Episodes that are completed
    
    ########


    files = sorted(files, key=str.casefold)
    for file in files:
        if file.endswith('.json.gz'):

            ######
            #TODO: Changed
            if file in ignore_scenes:
                print(f"Found Invalid Scene: {file}! Ignoring...")
                continue

            scenes_count += 1

            ######

            with gzip.open(os.path.join(object_nav_path, file), 'r') as f:
                json_data = json.load(f)
                scene_id = json_data['episodes'][0]['scene_id']

                #For a Scene ID, add objects and their locations
                if scene_id not in scene_data:
                    scene_data_ = SceneData(scene_id, {}, {})           #TODO: Gather individual scene info from here on
                    for obj_ in json_data['goals_by_category']:         #TODO: Add objects by category into the scene
                        obj = json_data['goals_by_category'][obj_]
                        obj_name = obj[0]['object_category']            #TODO: Get Object Name or Category
                        scene_data_.object_locations[
                            obj_name] = []  # the actual locations and bounding boxes will be loaded later
                        scene_data_.object_ids[obj_name] = []
                        for obj_loc in obj:                             #TODO: Add Object Locations
                            scene_data_.object_ids[obj_name].append(obj_loc['object_id'])
                    scene_data[scene_id] = scene_data_                  #TODO: Add gathered scene info to scene_data
                
                #Define episodes for Scene: Each episode corresponds to a goal object category, start position and rotation
                for ep in json_data['episodes']:
                    episode = Episode(ep['scene_id'],
                                      ep["episode_id"],    #   i,  #TODO Changed: i -> ep["episode_id"]
                                      ep['start_position'],
                                      ep['start_rotation'],
                                      [ep['object_category']],
                                      ep['info']['geodesic_distance'])
                    episodes.append(episode)
                    i += 1

    #####
    print(f"\nLoaded {scenes_count} scenes and {i} episodes.")
    #####
    return episodes, scene_data

def load_hm3d_objects(scene_data: Dict[str, SceneData], semantic_objects, scene_id: str):
    for scene_obj in semantic_objects:
        obj_name = scene_obj.category.name()
        for cat in scene_data[scene_id].object_locations.keys():
            if scene_obj.id in scene_data[scene_id].object_locations[cat]:
                continue
            if scene_obj.semantic_id in scene_data[scene_id].object_ids[cat]:
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
            elif obj_name in cat or cat in obj_name:
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
            elif cat == "plant" and ("flower" in obj_name):
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
            elif cat == "sofa" and ("couch" in obj_name):
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
    return scene_data


if __name__ == '__main__':
    eps, scene_data = load_hm3d_episodes([], {}, "datasets/objectnav_hm3d_v1/val/content")
    print(f"Found {len(eps)} episodes")
    scene_dist = {}
    for ep in eps:
        if ep.scene_id not in scene_dist:
            scene_dist[ep.scene_id] = 1
        else:
            scene_dist[ep.scene_id] += 1

    for sc in scene_dist:
        print(f"Scene {sc}, number of eps {scene_dist[sc]}")

    obj_counts = {}
    for ep in eps:
        for obj in ep.obj_sequence:
            if obj not in obj_counts:
                obj_counts[obj] = 1
            else:
                obj_counts[obj] += 1
    total = sum([obj_counts[obj] for obj in obj_counts])
    for obj in obj_counts:
        print(f"Object {obj}, count {obj_counts[obj]}, percentage {obj_counts[obj] / total}")