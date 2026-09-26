import numpy as np
from typing import Dict, List
import os.path as osp
import json


def cosine_sim(arr_1: np.ndarray, arr_2: np.ndarray) -> float:

    eps = 1e-8
    arr_1_norm = max(np.linalg.norm(arr_1), eps)
    arr_2_norm = max(np.linalg.norm(arr_2), eps)
    return (np.dot(arr_1, arr_2) / (( arr_1_norm * arr_2_norm )))


def arr_min_arg(arr):
    return np.unravel_index(np.argmin(np.ravel(arr)), shape = arr.shape)

def get_min_dist(true_arr, pred_arr):

    assert (true_arr.shape[1] == 2) and (pred_arr.shape[1] == 2), "Please provide arrays in shape (num_pts, 2)"

    dists = np.linalg.norm(true_arr[:, np.newaxis, :] - pred_arr[np.newaxis, :, :], axis = 2)
    return (np.min(dists), arr_min_arg(dists)[1])



def eval_preds(true_pos, pred_pos, 
               success_thresh = 1):
    
    min_dist = np.inf

    for ind in pred_pos.keys():
        min_dist_ind, pred_num_ind = get_min_dist(true_pos, pred_pos[ind])
        
        if min_dist_ind < min_dist:
            min_dist = min_dist_ind
            min_pred = pred_num_ind
            min_ind = ind
    
    success = int(min_dist <= success_thresh)
    return success, float(min_dist), min_pred, min_ind

    

    



def ovon_eval_metrics(results: Dict,
                   skipped_cats: List,
                   save_dir: str = None):

    # if "val_hssd" in results:
    #     val_dirs = ["val_hssd"]
    # else:
    #     # val_dirs = ['val_seen', 'val_unseen', 'val_seen_synonyms']
    #     val_dirs = ['val_seen_synonyms']

    val_dirs = [k for k in results.keys() if k.startswith("val_")]
    print(f"Metrics val dirs: {val_dirs}")
    top_k_preds = results["top_k_preds"]

    success = {val_dir: {f'k_{i}' : 0 for i in top_k_preds} for val_dir in val_dirs}
    min_dist = {val_dir: {f'k_{i}' : [] for i in top_k_preds} for val_dir in val_dirs}
    count_micro = {val_dir: 0 for val_dir in val_dirs}
    inv_cats = {val_dir : [] for val_dir in val_dirs}

    print(f"\nCalculating Micro Eval Metrics...")
    for val_dir in val_dirs:
        for scene in results[val_dir].keys():
            for scene_episode in results[val_dir][scene].keys():
                for cat in results[val_dir][scene][scene_episode].keys():

                    if cat in skipped_cats:
                        print(f"Skipping {cat}")
                        inv_cats[val_dir].append(cat)
                        continue
                    
                    try:
                        top_k_success = results[val_dir][scene][scene_episode][cat]['top_k_success']
                        top_k_min_dist = results[val_dir][scene][scene_episode][cat]['top_k_min_dist']
                    except:
                        print(f"Skipping category in evaluation: {cat}")
                        inv_cats[val_dir].append(cat)
                        continue

                    for k_i in top_k_success.keys():
                        success[val_dir][k_i] += top_k_success[k_i]
                        min_dist[val_dir][k_i].append(top_k_min_dist[k_i])
                    
                    count_micro[val_dir] += 1

    print(f"\nFinished iteratiing over {count_micro} instances")

    success_micro = {}
    avg_min_dist_micro = {}
    median_min_dist_micro = {}

    for val_dir in val_dirs:
        success_micro[val_dir] = {k: round((v / count_micro[val_dir]) * 100, 2) for k, v in success[val_dir].items()}

        avg_min_dist_micro[val_dir] = {k: round(sum(v)/count_micro[val_dir], 2) for k, v in min_dist[val_dir].items()}
        median_min_dist_micro[val_dir] = {k: round(float(np.median(v)), 2) for k, v in min_dist[val_dir].items()}

    print("Success Micro Results:\n")
    for val_dir, k_dict in success_micro.items(): 
        print(f"\n\n - {val_dir} : \n")
        for k, v in k_dict.items():
            print(f"   - {k} : {v}\n")

    print("\nInvalid Categories:\n")
    for val_dir, inv_list in inv_cats.items(): 
        print(f"\n\n - {val_dir} : {len(inv_list)} \n")

    print(f"\nCalculating Macro Eval Metrics...")

    # val_norm = {val_dir : (count[val_dir] / sum(count.values())) for val_dir in val_dirs}
    # weighted_mean = lambda vals_d, weights_d: sum( [ vals_d[k] * weights_d[k] for k in vals_d.keys()] )
    # inner_dict_vals = lambda d, in_key: {out_key : in_dict[in_key] for out_key, in_dict in d.items()}

    # for i in top_k_preds:

    #     success_macro = { f"k_{i}" : weighted_mean( inner_dict_vals(success, f"k_{i}"), val_norm ) }

    success_macro = {f'k_{i}' : 0 for i in top_k_preds}
    min_dist_macro = {f'k_{i}' : [] for i in top_k_preds}
    count_macro = 0


    for val_dir in val_dirs:
        for scene in results[val_dir].keys():
            for scene_episode in results[val_dir][scene].keys():
                for cat in results[val_dir][scene][scene_episode].keys():

                    if cat in skipped_cats:
                        print(f"Skipping {cat}")
                        continue
                    
                    try:
                        top_k_success = results[val_dir][scene][scene_episode][cat]['top_k_success']
                        top_k_min_dist = results[val_dir][scene][scene_episode][cat]['top_k_min_dist']
                    except:
                        print(f"Skipping category in evaluation: {cat}")
                        continue

                    for k_i in top_k_success.keys():
                        success_macro[k_i] += top_k_success[k_i]
                        min_dist_macro[k_i].append(top_k_min_dist[k_i])
                    
                    count_macro += 1

    print(f"Finished with {count_macro} instances")


    success_macro = {k: round((v / count_macro) * 100, 2) for k, v in success_macro.items()}

    avg_min_dist_macro = {k: round(sum(v)/count_macro, 2) for k, v in min_dist_macro.items()}
    median_min_dist_macro = {k: round(float(np.median(v)), 2) for k, v in min_dist_macro.items()}

    print("\nSuccess Macro Results:\n")
    for k, v in success_macro.items(): print(f" - {k} : {v}\n")

    if save_dir is not None:

        save_path = osp.join(save_dir, "eval_metrics.json")

        eval_metrics_results = {
            "success_micro": success_micro,
            "success_macro": success_macro,
            "avg_min_dist_micro": avg_min_dist_micro,
            "avg_min_dist_macro": avg_min_dist_macro,
            "median_min_dist_micro": median_min_dist_micro,
            "median_min_dist_macro": median_min_dist_macro,
            "count_micro": count_micro,
            "count_macro": count_macro,
            "inv_cats": inv_cats

        }

        with open(save_path, "w") as f:
            json.dump(eval_metrics_results, f, indent = 2)
        
        print(f"Saved Eval Metrics to : {save_path}")










    
