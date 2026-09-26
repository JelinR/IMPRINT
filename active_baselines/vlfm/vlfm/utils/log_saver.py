# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import json
import os
import time
from typing import Dict, Union
import numpy as np

def log_episode(episode_id: Union[str, int], scene_id: str, data: Dict) -> None:
    log_dir = os.environ["ZSOS_LOG_DIR"]
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    base = f"{episode_id}_{scene_id}.json"
    filename = os.path.join(log_dir, base)

    # Skip if the filename already exists AND it isn't empty
    if not (os.path.exists(filename) and os.path.getsize(filename) > 0):
        print(f"Logging episode {int(episode_id):04d} to {filename}")
        with open(filename, "w") as f:
            json.dump({"episode_id": episode_id, "scene_id": scene_id, **data}, f, indent=4)
    else:
        print(f"Failed to Log File! Log for scene: {scene_id}, episode: {int(episode_id)} already exists at : {filename}")


#TODO: Added for embed
def log_embed_dict(episode_id: Union[str, int], scene_id: str, 
                   model_name: str,
                   embed_dict: Dict, 
                   init_dict: Dict,
                   log_dir: str) -> None:
    
    log_embed_dir = os.path.join(log_dir, f"embed_dicts/{scene_id}")  #os.environ["ZSOS_LOG_DIR"]
    log_init_dir = os.path.join(log_dir, "init_dicts")

    os.makedirs(log_embed_dir, exist_ok=True)
    os.makedirs(log_init_dir, exist_ok=True)

    #Logging Init Dictionary
    # init_base_file = f"init_dict_model_{model_name}_scene_{scene_id}.npz"
    init_base_file = f"init_dict_model_{model_name}_scene_{scene_id}_episode_{episode_id}.npz"
    init_filename = os.path.join(log_init_dir, init_base_file)

    if not (os.path.exists(init_filename) and os.path.getsize(init_filename) > 0):
        print(f"Logging Init Info for (Scene, Episode): {scene_id, episode_id} to {init_filename}")

        init_dict['scene'] = scene_id
        init_dict['episode'] = episode_id

        np.savez(init_filename, **init_dict)
    else:
        print(f"Logging initial dictionary FAILED: File already exists for (Scene, Episode) : ({scene_id}, {episode_id})")


    #Logging Embedding Dictionary (for each floors)
    floor_names = list(embed_dict.keys())

    #If not string, then change type to string
    str_tuple_wrap = lambda var: str(tuple(var)) if type(var) is not str else var

    for floor in floor_names:
        embed_base_file = f"embed_dict_model_{model_name}_scene_{scene_id}_episode_{episode_id}.npz"
        embed_filename = os.path.join(log_embed_dir, embed_base_file)

        if not (os.path.exists(embed_filename) and os.path.getsize(embed_filename) > 0):
            print(f"Logging Embedding for (Scene, Floor): ({scene_id}, {floor})")
            np.savez(embed_filename, **{str_tuple_wrap(k):v for (k, v) in embed_dict[floor].items()})
        else:
            print(f"Logging Embedding FAILED for (Scene, Floor): ({scene_id}, {floor}), File already exists!")



    # embed_base_file = f"embed_dict_model_{model_name}_scene_{scene_id}_ep_{episode_id}.npz"
    # embed_filename = os.path.join(log_embed_dir, embed_base_file)

    # init_base_file = f"init_dict_model_{model_name}_scene_{scene_id}_ep_{episode_id}.npz"
    # init_filename = os.path.join(log_init_dir, init_base_file)

    # #If not string, then change type to string
    # str_tuple_wrap = lambda var: str(tuple(var)) if type(var) is not str else var

    # if not (os.path.exists(embed_filename) and os.path.getsize(embed_filename) > 0):
    #     print(f"Logging Embedding for (Scene, Episode): {scene_id, episode_id} to {embed_filename}")
    #     np.savez(embed_filename, **{str_tuple_wrap(k):v for (k, v) in embed_dict.items()})
        
    #     print(f"Logging Init Info for (Scene, Episode): {scene_id, episode_id} to {init_filename}")
    #     np.savez(init_filename, **init_dict)
    # else:
    #     print("LOGGING FAILED: Logged Embed Dict already exists. Not Logging current Embed Dict and Init Info.")


def is_evaluated(episode_id: Union[str, int], scene_id: str) -> bool:
    log_dir = os.environ["ZSOS_LOG_DIR"]
    base = f"{episode_id}_{scene_id}.json"
    filename = os.path.join(log_dir, base)

    # Return false if the directory doesn't exist
    if not os.path.exists(log_dir):
        return False

    # Delete any empty files that are older than 5 minutes
    for f in os.listdir(log_dir):
        try:
            if os.path.getsize(os.path.join(log_dir, f)) == 0 and (
                time.time() - os.path.getmtime(os.path.join(log_dir, f)) > 300
            ):
                os.remove(os.path.join(log_dir, f))
        except Exception:
            pass

    return os.path.exists(filename)
