import os
import numpy as np
import json
from tqdm import tqdm

from static_baselines.arguments import get_args
from static_baselines.sim_grid import Sim_Grid
from static_baselines.utils.metrics import eval_preds, ovon_eval_metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging


try:
    from vlfm.vlm.blip2_embed import BLIP2_Embed
except:
    print(f"Loading BLIP2_Embed FAILED. This is fine if you are using the SED model.")

try:
    from vlfm.vlm.sed import OpenClip_Embed
except:
    print(f"Loading OpenClip_Embed FAILED! This is fine if you are using the BLIP2 Model.")


try:
    from vlfm.vlm.clip_embed import CLIP_UnifiedEmbedder
except:
    print(f"Loading CLIP/SIGLIP Unified Embedded FAILED. This is fine if you are not using this.")

#Time Loading Sim Dict and Obtaining top-k vals
import time
import os


def main():

    args = get_args()

    #Load Model
    print(f"Loading Model {args.model_name} ...")
    if args.model_name == "BLIP2":  
        model = BLIP2_Embed()
    elif args.model_name in ["clip", "siglip"]: 
        model = CLIP_UnifiedEmbedder(backend = args.model_name)
    elif args.model_name == "SED": 
        model = OpenClip_Embed()
    else:
        raise ValueError(f"Invalid model name : {args.model_name}! Valid options : BLIP2, SED, clip, siglip")
    print("Finished Loading Model")

    #Define Arguments
    model_name = args.model_name
    from_one_map = args.from_one_map
    save_imgs = args.save_imgs
    scrape_imgs = args.scrape_imgs
    top_k_preds = args.top_k_preds
    top_n_patch_embeds = args.top_n_patch_embeds

    floor_low_lim = args.floor_low_lim
    floor_high_lim = args.floor_high_lim

    root_log_dir = args.log_dir
    cat_pos_dir = args.cat_pos_dir
    embed_dir = args.embed_dir
    scrape_data_dir = args.scrape_data_dir
    validation_splits = get_validation_splits(cat_pos_dir)

    vals_grid_size_px = args.grid_size_px
    vals_prompt_mode = args.prompt_mode
    vals_scrape_num = args.scrape_num
    vals_process_sim_mode = args.process_sim_mode
    vals_success_thresh = args.success_thresh

    if (floor_low_lim is None) and (floor_high_lim is None):
        print("\nWARNING: There is no floor filtering being done by default. " \
        "If needed, use floor_low_lim and floor_high_lim arguments.")

    #Iterate through argument gridsearch
    for grid_size_px in vals_grid_size_px:
        for prompt_mode in vals_prompt_mode:
            for scrape_num in vals_scrape_num:
                for process_sim_mode in vals_process_sim_mode:
                    for success_thresh in vals_success_thresh:

                        print(f"\n\nGrid Size px : {grid_size_px}")
                        print(f"Prompt Mode: {prompt_mode}")
                        print(f"Scrape Number: {scrape_num}")
                        print(f"Scrape Imgs: {scrape_imgs}")
                        print(f"Process Sim Mode: {process_sim_mode}")
                        print(f"Success Threshold: {success_thresh}")
                        print(f"Top N Patch Embeds: {top_n_patch_embeds}")

                        ### Initializes and Defines required variables before starting the current run

                        #Scrape images if mode is image or multi
                        if prompt_mode == "text": scrape_imgs = False
                        else: scrape_imgs = True
                        
                        #Define Log Directory for saving results and images
                        tail_log_dir = f"model_{model_name}/scrape_{scrape_num}/process_sim_{process_sim_mode}/prompt_{prompt_mode}/grid_{grid_size_px}_succ_{success_thresh}"
                        log_dir = os.path.join(os.getcwd(), root_log_dir, tail_log_dir)
                        # log_dir = os.path.join(os.getcwd(), root_log_dir, f"model_{model_name}/scrape_{scrape_num}/process_sim_{process_sim_mode}/prompt_{prompt_mode}/grid_{grid_size_px}_succ_{success_thresh}")
                        print(f"Scrape Number: {scrape_num}, Prompt Mode: {prompt_mode}")
                        print(f"root log dir: {root_log_dir}")
                        print(f"tail log dir: {tail_log_dir}")
                        print(f"Log Dir: {log_dir}")
                        os.makedirs(log_dir, exist_ok=True)

                        #assert "eval_metrics.json" not in os.listdir(log_dir), f"Evaluated Metrics already exists at: {log_dir}"
                        if "eval_metrics.json" in os.listdir(log_dir):
                            print(f"\nEvaluated metrics already exists at {log_dir}. Skipping Iteration...\n")
                            continue

                        if save_imgs: 
                            img_log_dir = os.path.join(log_dir, 'sim_imgs')
                            os.makedirs(img_log_dir, exist_ok = True)

                        print(f"Log Dir: {log_dir}")
                        print(f"Scrape Images: {scrape_imgs}")

                        # Log arguments
                        log_file = os.path.join(log_dir, "args.log")
                        logging.basicConfig(filename = log_file, level=logging.INFO)
                        logging.info(f"\nParsed arguments: {vars(args)}")
                        print(f"Logged Arguments at: {log_file}\n")

                        #Obtain Embed directory, along with the valid scene names
                        embed_dir = os.path.join(args.embed_dir, f"grid_{grid_size_px}")
                        scene_names = [f for f in os.listdir(os.path.join(embed_dir, "embed_dicts"))]
                        print(f"Scene Names: \n{scene_names}")

                        #Dictionary with mapped scene and corresponding episodes
                        #Each episode is supposed to correspond with a unique floor mapping of the scene
                        eps_id_from_file_name = lambda file_name: str( file_name.split("episode_")[-1].split(".")[0] )        #TODO CHECK: Compatibility with old and new versions
                        # eps_id_from_file_name = lambda file_name: str( file_name.split("ep_")[-1].split(".")[0] )
                        scene_to_eps = {scene_name : [eps_id_from_file_name(file) for file in os.listdir(os.path.join(embed_dir, "embed_dicts", scene_name))] for scene_name in scene_names}
                        print(f"\nScene to Episodes:\n {scene_to_eps}")

                        #Ignore invalid scenes
                        invalid_scenes_path = os.path.join(embed_dir, "ignore_scenes.txt")
                        if os.path.exists(invalid_scenes_path):
                            with open(invalid_scenes_path, "r") as f:
                                invalid_scenes = f.readlines()

                            invalid_scenes = [name.strip() for name in invalid_scenes]

                            scene_names = [name for name in scene_names if name not in invalid_scenes]
                            print(f"\n!---Ignoring Scenes: {invalid_scenes}")
                        print(f"\nConsidering {len(scene_names)} scenes...")
                        
                        #Initialize Sim Grid
                        print(f"\nInitializing Sim Grid")
                        sim_grid = Sim_Grid(embeds_dir = embed_dir, 
                                            model_name = model_name,
                                            use_model = model) 


                        #Initialize results dict to save the results
                        #If temporary results dict exists, load it. Used to checkpoint random breaks in the runs.
                        temp_file_name = os.path.join(log_dir, f"{model_name}_temp.json")
                        if os.path.exists(temp_file_name):
                            with open(temp_file_name, "r") as f:
                                results_dict = json.load(f)
                            print(f"\n\nLoaded Results Dict from Saved Temp File...\n\n")

                        else:
                            results_dict = {
                                "model": model_name,
                                "prompt_mode": prompt_mode,
                                "grid_size_px": grid_size_px,
                                "success_thresh": success_thresh,
                                "top_k_preds": top_k_preds,
                                **{split: {} for split in validation_splits},
                            }
                        
                        #Define the val_set to iterate over. Obtained by using defined results_dict.
                        skipped_cats = []
                        top_k_vals = top_k_preds
                        val_sets = [val_key for val_key in results_dict.keys() if val_key.startswith("val_")]

                        #Iterating over the val_sets
                        for val_dir in val_sets:
                            print(f"\n--- Evaluating {val_dir} ---\n")
                            
                            #Iterating over each scene in the val_set
                            for scene in tqdm(scene_names):

                                if scene not in results_dict[val_dir].keys():
                                    results_dict[val_dir][scene] = {}

                                #Obtain mapped episodes for current scene. 
                                # Recall each mapped episode corresponds to unique floor mappings
                                scene_episodes = scene_to_eps[scene]

                                #Iterating over each mapped episode for the current scene
                                for scene_episode in scene_episodes:

                                    if scene_episode not in results_dict[val_dir][scene].keys():
                                        results_dict[val_dir][scene][scene_episode] = {}

                                    #Loads the embed and init dictionaries for this scene, episode (floor)
                                    sim_grid.load_embed_init(scene, int(scene_episode))

                                    if save_imgs:
                                        scene_log_dir = os.path.join(img_log_dir, f'{val_dir}/{scene}')
                                        os.makedirs(scene_log_dir, exist_ok=True)

                                    #Load dict for scene with goal category to habitat positions
                                    scene_cat_to_pos = load_scene_cat_to_pos(scene_name=scene,
                                                                                cat_pos_dir = cat_pos_dir,
                                                                                val_dir = val_dir)
                                    
                                    #Filter positions that are in other floors using fixed thresholds. Only keep valid goal positions.
                                    if (floor_low_lim is not None and floor_high_lim is not None):
                                        print(f"Filtering by height: [{floor_low_lim}, {floor_high_lim}]")
                                        scene_cat_to_pos = filter_by_height(scene_dict = scene_cat_to_pos,
                                                                            sim_grid = sim_grid,
                                                                            low_lim = floor_low_lim,
                                                                            up_lim = floor_high_lim)
                                    
                                    #For each valid goal category in this scene and episode (floor), 
                                    # 1. Use the required prompt query and obtain the similarity grid
                                    # 2. Obtain the top-k preds using the similarity grid
                                    # 3. Evaluate the metrics (SR, DTG) using the true px, pred px
                                    # 4. Update the results dictionary
                                    for cat, cat_pos in scene_cat_to_pos.items():
                                        
                                        #If a category is previously determined as invalid (images not found), then skip
                                        if cat in skipped_cats:
                                            if args.verbose: print(f"\n\n---Skipping Category: {cat}\n\n")
                                            continue

                                        #Defines cat_descr : A mapping from a numeric categoric ID to category name
                                        if "goat_descr" in cat_pos_dir:    #GOAT Dataset
                                            cat_descr_path = os.path.join(cat_pos_dir, "cat_to_descr.json")
                                            assert os.path.exists(cat_descr_path)

                                            with open(cat_descr_path, "r") as f:
                                                cat_descr = json.load(f)
                                        else:
                                            cat_descr = None

                                        #If a category is already evaluated for this scene and episode, then skip
                                        print(f"\n\n------Category: {cat}")
                                        if cat in results_dict[val_dir][scene][scene_episode].keys(): 
                                            print(f"Category {cat} already evaluated. Skipping...")
                                            continue
                                        else:
                                            results_dict[val_dir][scene][scene_episode][cat] = {}
                                        
                                        
                                        #Loading Similarity dictionary using the prompt query.
                                        #Failure to load can be due to failure to retrieve relevant images
                                        try:
                                            sim_grid.load_sim_dict(text = cat, 
                                                                    prompt_mode = prompt_mode,
                                                                    process_sim_mode = process_sim_mode,
                                                                    scrape_imgs = scrape_imgs,
                                                                    scrape_num = scrape_num,
                                                                    scrape_data_dir = scrape_data_dir,
                                                                    cat_descr = cat_descr,
                                                                    top_n_patch_embeds = top_n_patch_embeds   )
                                        except:
                                            print(f"\n\n---Error in loading Sim Dict. Possible failure due to not being able to retrieve relevant images.\nSkipping Category: {cat}\n\n")
                                            skipped_cats.append(cat)
                                            continue


                                        #Obtain the pixel coordinates of the object category
                                        cat_pos = cat_pos[:, [0, 2]]
                                        cat_px = sim_grid.hab_to_px(hab_pts = cat_pos, from_one_map = from_one_map)

                                        #Save the Sim Grid
                                        if save_imgs:
                                            sim_grid.visualize(save_dir = os.path.join(scene_log_dir, f'{scene}_{model_name}_{cat}.png'),
                                                            plot_pts = cat_px,
                                                            point_size = args.plot_point_size)

                                        top_k_success = {f"k_{i}" : 0 for i in top_k_vals}
                                        top_k_min_dist = {f"k_{i}" : 0 for i in top_k_vals}
                                        # top_k_pred_freq = {f"k_{i}" : 0 for i in top_k_vals}

                                        #Obtains the top-k evaluations (top-1, top-3, top-5,...)
                                        for k in top_k_vals:
                                            
                                            pred_px = sim_grid.top_k_sims(k = k)
                                            pred_pos = {ind : sim_grid.px_to_hab(top_preds_px, from_one_map) for ind, top_preds_px in pred_px.items()}

                                            success, min_dist, nth_pred, min_input = eval_preds(true_pos = cat_pos,
                                                                                    pred_pos = pred_pos,
                                                                                    success_thresh = success_thresh)
                                            
                                            top_k_success[f"k_{k}"] = success
                                            top_k_min_dist[f"k_{k}"] = min_dist

                                        results_dict[val_dir][scene][scene_episode][cat] = {
                                                                            'top_k_success': top_k_success,
                                                                            'top_k_min_dist': top_k_min_dist
                                        }

                                        #Temporary save for results (in case of server disconnect)
                                        with open(temp_file_name, "w") as f:
                                            json.dump(results_dict, f)
                                            print(f"Saved temp file at : {temp_file_name}")

                        #If any skipped categories, save it
                        if len(skipped_cats) > 0:
                            print(f"Skipped Categories: {skipped_cats}")
                            txt_file_name = os.path.join(log_dir, f"skipped_cats.txt")
                            with open(txt_file_name, 'w') as f:
                                f.write(",".join(skipped_cats))
                            print(f"Saved Skipped Cats")

                        #Save Results into a JSON file
                        json_file_name = os.path.join(log_dir, f'{model_name}_static_evals.json')
                        with open(json_file_name, 'w') as f:
                            json.dump(results_dict, f, indent=2)
                        print(f"Saved Evaluated JSON file to : {json_file_name}")


                        #Calculating Success Metrics from results
                        ovon_eval_metrics(results_dict, skipped_cats,
                                        save_dir = log_dir)
                        


def sanitize_filename(name):
    """Sanitize filenames to avoid illegal characters."""
    return re.sub(r'[<>:"/\\|?*]', '', name).strip().replace("\r", "").replace("\n", "").rstrip(".")


def get_validation_splits(cat_pos_dir):
    """Return all static validation splits available for a category-position directory."""
    if "hssd" in cat_pos_dir.lower():
        return ["val_hssd"]

    splits = [
        name
        for name in os.listdir(cat_pos_dir)
        if name.startswith("val_") and os.path.isdir(os.path.join(cat_pos_dir, name))
    ]
    if not splits:
        raise FileNotFoundError(f"No validation splits found in {cat_pos_dir}")
    return sorted(splits)

def load_scene_cat_to_pos(scene_name, cat_pos_dir, val_dir):

    if "hssd" in cat_pos_dir:
        file_path = os.path.join(cat_pos_dir, f"{scene_name}.npy")
    else:
        file_path = os.path.join(cat_pos_dir, f'{val_dir}/{scene_name}_{val_dir}.npy')

    goals_pos = np.load(file_path, allow_pickle=True)
    goals_pos = dict(goals_pos.item())
    return goals_pos

def filter_by_height(scene_dict, sim_grid, low_lim = 0.5, up_lim = 1.5):

    hab_z = sim_grid.init_dict['init_pose'].item()['init_pos_abs'][1]
    hab_z_lims = (hab_z - low_lim, hab_z + up_lim)

    filter_pos = lambda pos: pos[ (pos[:, 1] <= hab_z_lims[1]) & (pos[:, 1] >= hab_z_lims[0]) ]

    scene_filt = {goal : filter_pos(pos) for goal, pos in scene_dict.items()}
    scene_filt = {goal : pos for goal, pos in scene_filt.items() if len(pos) > 0}

    return scene_filt


if __name__ == "__main__":
    main()
