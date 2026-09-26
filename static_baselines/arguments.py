import argparse
import os

def get_args():

    parser = argparse.ArgumentParser(
        description="VLFM-Query-Evaluation"
    )

    parser.add_argument("--model_name", type=str, default="BLIP2")
    parser.add_argument("--from_one_map", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--grid_size_px", type=int, nargs="+", default=[20], help="Size (in pixels) for grid cell.")
    
    #Visualization Args
    parser.add_argument("--save_imgs", type=bool, default=False, help="Save similarity grid with true and pred positions")
    parser.add_argument("--plot_point_size", type=int, default=50, help="Size of scatter points in the plot")
    
    #Similarity Dictionary Args
    parser.add_argument("--prompt_mode", type=str, nargs="+", default=["text"])
    parser.add_argument("--scrape_imgs", type=bool, default=True)
    parser.add_argument("--scrape_num", type=int, nargs="+", default=[15], help="Number of images to scrape for a given category or prompt.")
    parser.add_argument("--top_k_preds", type=int, nargs=3, default=[1, 3, 5], help="Number of Top Predictions to consider. Provide a list of 3 values.")
    parser.add_argument("--process_sim_mode", type=str, nargs="+", default=["mean"], help="Process Similarity Dictionary for multiple inputs using mean or harmonic mean")
    parser.add_argument("--success_thresh", type=float, nargs="+", default=[1.0], help="Minimum Distance to classify as success.")
    parser.add_argument("--top_n_patch_embeds", type=int, default=20, help="Top-n Patch Embeddings to consider when obtaining SED Image Embedding using Text Embedding.")
    
    #Filter Floor Args
    parser.add_argument("--floor_low_lim", type=float, default=None, help="How much lower than Agent would the current floor be at.")
    parser.add_argument("--floor_high_lim", type=float, default=None, help="How much higher than Agent would the current floor be at.")
    
    #Data and Results Args
    parser.add_argument("--log_dir", type=str, default="static_baselines/eval_results", help="Directory for logging results")
    parser.add_argument("--cat_pos_dir", type=str, default="static_baselines/cat_pos/ovon_cat_pos", help="OVON-syn category-position data")
    parser.add_argument("--embed_dir", type=str, default="static_baselines/static_embeddings", help="Directory with static scene embeddings")
    parser.add_argument("--scrape_data_dir", type=str, default="static_baselines/data/scraped_imgs", help="Directory with reference images")




    #Parse arguments
    args = parser.parse_args()
    
    if args.prompt_mode == "text": args.scrape_imgs = False
    if args.process_sim_mode == "None": args.process_sim_mode = None
    
    return args