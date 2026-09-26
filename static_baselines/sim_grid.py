import cv2
import numpy as np
from typing import Dict, Any, List
import ast
import heapq
import os
from functools import partial
from scipy.spatial.transform import Rotation as R
import torch

from matplotlib import cm, colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import vlfm.utils.embed_utils as eu
import vlfm.utils.geometry_utils as gu
from static_baselines.utils.scrape_img import load_images
from static_baselines.utils.metrics import cosine_sim

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



ROOT_DIR = 'embed_results/grid_4'
DEBUG = True


class Sim_Grid:

    def __init__(self, embeds_dir: str = ROOT_DIR, 
                 model_name: str = "BLIP2", use_model = None):

        self.results_dir = embeds_dir

        assert model_name in ["BLIP2", "SED", "clip", "siglip"], "Model name should be : ['BLIP2', 'SED', 'clip', 'siglip']"                  
        
        self.model_name = model_name
        # print(f"Loading {self.model_name} model...")

        if use_model is not None:
            self.model = use_model
            print(f"Using already loaded {self.model_name} model...")
        elif model_name == "BLIP2":
            print(f"Loading {self.model_name} model...")
            self.model = BLIP2_Embed()
        elif model_name in ['clip', 'siglip']:
            print(f"Loading {self.model_name} model...")
            self.model = CLIP_UnifiedEmbedder(backend = model_name)
        else:
            print(f"Loading {self.model_name} model...")
            self.model = OpenClip_Embed()


        self.sim_dict = {}
        self.text, self.img = None, None
        self.prompt_mode = None

        self.scene_name = None
        self.episode_id = None
        self.init_dict, self.embed_dict = None, None
        self.grid_size = None
        self.tf_hab_to_agent = None


    def load_embed_init(self,
                        scene_name: str,
                        episode_id: int = None):
        r"Loads the model (if not loaded), transformation matrix, embedding and init dicts"
        
        self.scene_name = scene_name     

        print(f"Scene name: {scene_name}, Episode: {episode_id}")

        self.init_dict = self.load_dict('init', episode_id)
        self.embed_dict = self.load_dict('embed', episode_id)

        print(self.init_dict)

        self.episode_id = int(self.init_dict["episode"]) if episode_id is None else episode_id

        print(f"Loaded Embed and Init dicts for Scene: {self.scene_name}, Episode {episode_id} and Model: {self.model_name}")

        #Tranformation matrix from habitat to agent coordinates
        hab_init_pose = self.init_dict['init_pose'].item()
        self.tf_hab_to_agent = self.load_tf_hab_to_agent(hab_init_pose)

        try:
            self.grid_size = self.init_dict['grid_cell_size']
        except KeyError:
            self.grid_size = int(self.init_dict['pixels_per_meter'])
            print(f" Grid Size not found in init dict keys. \n Assuming grid size to be 1 m.sq : {self.grid_size} pixels")
            self.init_dict["grid_cell_size"] = self.init_dict["pixels_per_meter"]

    def load_sim_dict(self, imgs: List[np.ndarray] = None, text: str = None, 
                      prompt_mode: str = "text", process_sim_mode: str = None,
                      scrape_imgs: bool = False, scrape_num: int = 3, scrape_data_dir: str = "data/scraped_imgs/ovon_15",
                      cat_descr: Dict = None, top_n_patch_embeds: int = 1):
        r"""Loads similarity dictionary with grid points and corresponding cosine similarity scores. These 
        are calculated in three modes: 
            - image: Uses only image prompt. Can be given as input, or can be retrieved from internet using scrape_imgs arg.
            - text: Uses only text prompt. 
            - multi: Uses both text and image embeds, and combines them in certain ways to produce the scores.        
        """

        print(f"\nPrompt Mode: {prompt_mode}\nProcess Sim Mode: {process_sim_mode}\nScrape Num: {scrape_num}\n")

        assert self.embed_dict is not None, "Embeddings are not loaded yet! Please run load_embed_init method first."

        if scrape_imgs:
            assert text is not None, "Please provide a text prompt to scrape for images."

            if cat_descr is not None:
                main_cat = text.split(",")[0]
                sub_cat = cat_descr[main_cat][text]

                query = sub_cat
                search_dir = os.path.join(scrape_data_dir, main_cat)
            else:
                query = text
                search_dir = scrape_data_dir

            

            imgs = load_images(query = query, num_images = scrape_num, save_dir = search_dir)
            assert len(imgs) > 0, "Number of scraped images is zero!"

        self.imgs = imgs
        self.text = text
        self.prompt_mode = prompt_mode

        if self.prompt_mode == "text": self.imgs = None
        # elif self.prompt_mode == "image": self.text = None
        elif self.prompt_mode == "multi":
            assert (self.imgs is not None) and (self.text is not None)
        else:
            assert self.prompt_mode in ["text", "image", "multi"], "Please provide a valid input for prompt mode: image, text, multi"

        multi_target_embeds = self.get_embed_eval(top_n_patch_embeds=top_n_patch_embeds) 

        # self.sim_dict = {px : cosine_sim(np.array(target_embed), np.array(px_embed)) for px, px_embed in self.embed_dict.items()}

        self.sim_dict = {}
        for input_ind in range(multi_target_embeds.shape[0]):
            target_embed = multi_target_embeds[input_ind]
            self.sim_dict[input_ind] = {px : cosine_sim(np.array(target_embed), np.array(px_embed)) for px, px_embed in self.embed_dict.items()}

        if process_sim_mode is not None and len(self.sim_dict) > 1:
            self.process_sim_dict(process_mode = process_sim_mode)

    def load_dict(self,
                  dict_type: str,
                  episode_id: int = None):
        r"Loads embed or init dict from the embeds_dir corresponding to the current scene and model name"
        
        assert dict_type in ['embed', 'init']

        dict_dir = os.path.join(self.results_dir, f'{dict_type}_dicts')

        # if dict_type == "embed": 
        #     dict_dir = os.path.join(dict_dir, f'{self.scene_name}')
        #     # file_name = f'{dict_type}_dict_model_{self.model_name}_scene_{self.scene_name}'

        #     valid_files = [f for f in os.listdir(dict_dir) if f.__contains__(file_name)]      #TODO Changed
        #     # valid_files = [f for f in os.listdir(dict_dir) if (f.__contains__(self.scene_name) and f.__contains__(dict_type) and f.__contains__(self.model_name))]
        #     #valid_files = [f for f in os.listdir(dict_dir) if (f.__contains__(self.scene_name) and f.__contains__(dict_type))]
        #     if len(valid_files) > 1:
        #         print(f'There are more than 1 files : {len(valid_files)} files found . Considering the first file...')

        #     dict_path = os.path.join(dict_dir, valid_files[0])
        
        # else:
        #     file_name = f'{dict_type}_dict_model_{self.model_name}_scene_{self.scene_name}.npz'
        #     dict_path = os.path.join(dict_dir, file_name)

        if dict_type == "embed":
            dict_dir = os.path.join(dict_dir, f'{self.scene_name}')
        # file_name = f'{dict_type}_dict_model_{self.model_name}_scene_{self.scene_name}'

        #valid_files = [f for f in os.listdir(dict_dir) if f.__contains__(file_name)]      #TODO Changed
        #valid_files = [f for f in os.listdir(dict_dir) if (f.__contains__(self.scene_name) and f.__contains__(dict_type))]
        #valid_files = [f for f in os.listdir(dict_dir) if (f.__contains__(self.scene_name) and f.__contains__(dict_type) and f.__contains__(self.model_name))]
        valid_files = [f for f in os.listdir(dict_dir) \
                       if (f.__contains__(self.scene_name) and \
                           f.__contains__(dict_type) and \
                            f.__contains__(self.model_name))]

        # print(dict_dir, self.scene_name, dict_type, self.model_name)
        # print(valid_files)
        
        if episode_id is not None:
        # if (episode_id is not None) and (dict_type == "embed"):
            # print(f"Considering Episode ID: {episode_id} for Scene: {self.scene_name}")
            valid_files = [f for f in valid_files if (f.__contains__(f"episode_{episode_id}"))]   #TODO CHECK: Compatilibility with versions
            # valid_files = [f for f in valid_files if (f.__contains__(f"ep_{episode_id}"))]
        
        if len(valid_files) > 1:
            print(f'There are more than 1 files : {len(valid_files)} files found . Considering the first file...')

        dict_path = os.path.join(dict_dir, valid_files[0])
        print(f"Dict path: {dict_path}")

        # file_name = f'{dict_type}_dict_model_{self.model_name}_scene_{self.scene_name}'

        # valid_files = [f for f in os.listdir(dict_dir) if f.__contains__(file_name)]
        # if len(valid_files) > 1:
        #     print(f'There are more than 1 files : {len(valid_files)} files found . Considering the first file...')
        
        # dict_path = os.path.join(dict_dir, valid_files[0])


        dict_file = np.load(dict_path, allow_pickle=True)

        if dict_type == 'embed':
            if self.model_name in ['BLIP2', 'clip', 'siglip']:
                return {eval(k) : v.squeeze(0) for k, v in dict_file.items()}
            else:
                return {eval(k) : v for k, v in dict_file.items()}


        return dict(dict_file)
        
    def get_embed_eval(self, top_n_patch_embeds = 1):

        #If text embed is required
        if not (self.prompt_mode == "image"):
            multi_embeds = self.model.get_embed(txt = self.text).cpu().numpy()
        
        #If image embed is required
        if not (self.prompt_mode == "text"):
            
            for img_count, img in enumerate(self.imgs):
                
                if img_count == 0 and self.prompt_mode == "image":

                    if self.model_name == "SED":
                        multi_embeds = self.model.get_embed(image = img, txt = self.text, top_n_patch_embeds = top_n_patch_embeds).cpu().numpy()
                    else:
                        multi_embeds = self.model.get_embed(image = img).cpu().numpy()
                    continue

                    # multi_embeds = self.model.get_embed(image = img, 
                    #                                     txt = self.text if self.model_name == "SED" else None).cpu().numpy()
                    # continue

                if self.model_name == "SED":
                    img_embed = self.model.get_embed(image = img, txt = self.text, top_n_patch_embeds = top_n_patch_embeds).cpu().numpy()
                else:
                    img_embed = self.model.get_embed(image = img).cpu().numpy()
                # img_embed = self.model.get_embed(image = img,
                #                                  txt = self.text if self.model_name == "SED" else None).cpu().numpy()     #Of shape (1, 768)
                multi_embeds = np.concatenate((multi_embeds, img_embed), axis = 0) #Of Shape (num_imgs (+ 1), 768)

        return multi_embeds

    def process_sim_dict(self, process_mode):
        
        assert process_mode in ["mean", "harmonic_mean", "multi_mean"]

        #Gather all the values corresponding to each grid pixel point
        px_pts = self.sim_dict[next(iter(self.sim_dict))].keys()
        values_dict = {px : np.array([sub_sim_dict[px] for sub_sim_dict in self.sim_dict.values()]) for px in px_pts}
        # print(f"PX value shape: {values_dict[next(iter(values_dict))].shape}, {values_dict[next(iter(values_dict))]}")
        
        if process_mode == "mean":
            print(f"Processing Similarity Dictionary: Arithmetic Mean")
            mean_dict = {px : np.mean(vals) for px, vals in values_dict.items()}
            self.sim_dict = {0 : mean_dict}
        
        if process_mode == "harmonic_mean":
            print("Processing Similarity Dictionary: Harmonic Mean")
            harm_mean_dict = {px : len(vals) / np.sum(1.0 / vals) for px, vals in values_dict.items()}
            self.sim_dict = {0 : harm_mean_dict}

        if process_mode == "multi_mean": #Harmonic_Mean(Text, Mean(Images))
            print("Processing Similarity Dictionary: Multi Mean")
            multi_mean_dict  = {px : np.array([arr[0], np.mean(arr[1:])]) for px, arr in values_dict.items()}
            # print(f"MEAN PX value shape: {multi_mean_dict[next(iter(multi_mean_dict))].shape}, {multi_mean_dict[next(iter(multi_mean_dict))]}")
            multi_mean_dict = {px : len(vals) / np.sum(1.0 / vals) for px, vals in multi_mean_dict.items()}
            self.sim_dict = {0 : multi_mean_dict}



    def top_k_sims(self, k: int, return_sims: bool = False):
        r"""
        Returns top k pixel positions and their similarity scores
        """

        top_px, top_sims = {}, {}
        for input_ind in self.sim_dict.keys():
            top_items = heapq.nlargest(n = k, 
                                iterable = self.sim_dict[input_ind].items(), 
                                key = lambda item: item[1])
            
            top_px[input_ind] = np.array([list(item[0]) for item in top_items])
            top_sims[input_ind] = np.array([item[1] for item in top_items])

        if not return_sims:  return top_px
        return top_px, top_sims
        

    def load_tf_hab_to_agent(self, init_pose):
        r"Transforms Habitat coordinates to Agent coordinates"

        #Get x, y coords
        x, y = np.array(init_pose['init_pos_abs'])[[0, 2]]
        robot_xy = np.array([-x, -y, 0])

        #Get yaw
        quat = init_pose['init_rot_abs']
        quat = [quat[0], quat[2], quat[1], quat[3]]
        rot = R.from_quat(quat).as_rotvec(degrees = False)
        robot_yaw = rot[2]
        
        #First translate, then rotate
        tf_mat_trans = gu.xyz_yaw_to_tf_matrix(xyz = robot_xy, yaw = 0)
        tf_mat_rot = gu.xyz_yaw_to_tf_matrix(xyz = np.zeros_like(robot_xy), yaw = robot_yaw)

        tf_mat = tf_mat_rot @ tf_mat_trans

        return tf_mat

    def px_to_hab(self, px_pts, from_one_map=False):
        r"""
        Converts from pixels to habitat coordinates
        Pipeline:
            -> Swap pixel coords (accounting for using cv2 contours)
            -> Convert pixels to agent coordinates
                -> Inverts y axis 
                -> Recenters to pixel origin 
                -> Rescales pixels to meters
                -> Swaps the x and y positions: (y, x) -> (x, y)
            -> Convert agent coords to habitat coords
        """

        if from_one_map:
            print(f"\n!! OneMap Conversion : Pixel to Hab\n")

            px = px_pts.copy()

            px[:, 0] = (px[:, 0] - self.init_dict["pixel_origin"][0]) * self.grid_size * -1
            px[:, 1] = (px[:, 1] - self.init_dict["pixel_origin"][1]) * self.grid_size * -1

            px = px[:, ::-1]
            return px


        px = px_pts.copy()
        px = px[:, ::-1]

        #Pixels to Agent Coords
        px[:, 0] = self.init_dict['map_shape'][0] - px[:, 0]
        
        xy_pts = (px - self.init_dict['pixel_origin']) / self.init_dict['pixels_per_meter']
        xy_pts = xy_pts[:, ::-1]

        #Agent to Hab Coords
        tf_mat_inv = np.linalg.inv(self.tf_hab_to_agent)
        xy_pts = np.hstack((xy_pts, np.zeros((xy_pts.shape[0], 1))))
        hab_pts = list(map(lambda pt: gu.transform_points(tf_mat_inv, pt[np.newaxis, :]), xy_pts))
        hab_pts = np.array(hab_pts).squeeze(1)[:, :2]

        return hab_pts
    
    def hab_to_px(self, hab_pts, from_one_map=False):
        r"""
        Converts from habitat coords to pixel coordinates
        Pipeline:
            -> Convert habitat to agent coordinates
            -> Convert agent to pixel coordinates
                -> Swap coords
                -> Rescales to pixel scale, and recenters to pixel origin
                -> Inverts the y axis
            -> Swap pixel coords
        """

        if from_one_map:
            print(f"\n!! OneMap Conversion : Pixel to Hab\n")

            epsilon = 1e-9

            px_pts = hab_pts.copy()
            px_pts = px_pts[:, ::-1] * -1

            px_pts[:, 0] = (px_pts[:, 0] / self.grid_size + self.init_dict["pixel_origin"][0] + epsilon).astype(int)
            px_pts[:, 1] = (px_pts[:, 1] / self.grid_size + self.init_dict["pixel_origin"][1] + epsilon).astype(int)

            return px_pts

        #Habitat to Agent coords
        hab = np.hstack((hab_pts, np.zeros((hab_pts.shape[0], 1))))
        xy = list(map(lambda pt: gu.transform_points(self.tf_hab_to_agent, pt[np.newaxis, :]), hab))
        xy = np.array(xy).squeeze(1)[:, :2]
        
        #Agent to Pixel coords
        xy = xy[:, ::-1]
        px = np.rint(xy * self.init_dict['pixels_per_meter']) + self.init_dict['pixel_origin']
        px[:, 0] = self.init_dict['map_shape'][0] - px[:, 0]

        return px.astype(int)[:, ::-1]
    
    def px_to_arr(self, px_pts, arr_origin):
        r"""
        Returns pixels values accounting for the grid size and array origin
        """

        px = px_pts.copy()
        px = px[:, ::-1]

        px[:, 0] = self.init_dict['map_shape'][0] - px[:, 0]
        px_rel = (px - self.init_dict['pixel_origin']) / self.init_dict['pixels_per_meter']
        px_rel = px_rel[:, ::-1]

        arr_pts = px_rel + arr_origin

        return arr_pts
    
    def load_embed_np_arr(self, visualize=False):
        r"""
        Creates a numpy array with embeddings at relevant positions (zeros at other positions).
        This is supposed to represent the map accounting for the grid size.
        E.g.
            -> Map of size 1000x1000
            -> Grid size of 20x20
            -> Embedding dimension of 768
            Creates an array of shape (50, 50, 768)
        """
        
        arr_shape = self.init_dict['map_shape'] // self.grid_size
        arr_origin = arr_shape // 2
        embed_dim = next(iter(self.embed_dict.values())).shape[0]

        embed_arr = np.zeros((arr_shape[0], arr_shape[1], embed_dim))
        print(f'Loading array of shape: {embed_arr.shape}')
        
        px_to_arr_pos = {px : self.px_to_arr(np.array([px]), arr_origin)[0] \
                         for px in self.embed_dict.keys()}
        
        for px_pos in self.embed_dict.keys():
            arr_pos = px_to_arr_pos[px_pos]
            embed_arr[int(arr_pos[0]), int(arr_pos[1])] = self.embed_dict[px_pos]

        if visualize:
            vis_arr = embed_arr.sum(2)
            vis_arr[vis_arr > 0] = 1
            plt.imshow(vis_arr)
        
        return embed_arr


    def visualize(self, save_dir = None, plot_pts = None, 
                  plot_grid: bool = True, plot_reorient: bool = False,
                  point_size: int = 10):

        assert len(self.sim_dict) > 0, 'Please load the sim dict first'


        fig, ax = plt.subplots(figsize = (8, 8))

        top_pts = self.top_k_sims(k = 4)[0]
        if plot_reorient: top_pts = self.rot_pts(top_pts)

        ax.scatter(top_pts[:, 0], top_pts[:, 1], 
                        facecolors='white', edgecolors='black', 
                        s=150, linewidths=2.5)
        
            
        self.plot_scores_in_grid(ax = ax, fig = fig, 
                                    plot_grid = plot_grid,
                                    plot_reorient=plot_reorient,
                                    point_size=point_size)

        if plot_pts is not None:
            if plot_reorient: plot_pts = self.rot_pts(plot_pts)
            ax.scatter(plot_pts[:, 0], plot_pts[:, 1], 
                        facecolors='white', edgecolors='green', 
                        s=150)
            ax.scatter(plot_pts[:, 0], plot_pts[:, 1], color = 'black')



        text_add = f'Prompt Mode: Multi_Modal\nText Prompt: {self.text}' if self.prompt_mode == "multi" \
                    else f'Prompt Mode: Text\nText Prompt: {self.text}' if self.prompt_mode == "text" \
                    else f'Prompt Mode: Image'
        
        fig.suptitle(f'Scene: {self.scene_name}, Model: {self.model_name}, {text_add}\n', 
                     fontstyle='italic', fontsize = 15)
        fig.tight_layout()

        if save_dir is not None:
            if not save_dir.endswith('png'):
                save_dir = os.path.join(save_dir, f'scene_{self.scene_name}_model_{self.model_name}.png')
            fig.savefig(save_dir)

            plt.close(fig)
        
        else: plt.show()

    def plot_img(self, ax: plt.Axes, title: str = 'Image Prompt'):
        ax.imshow(self.img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        
    def plot_scores_in_grid(self, ax: plt.Axes, fig: plt.figure, 
                            plot_grid: bool = False, plot_reorient: bool = False,
                            point_size: int = 10):

        #Get grid points and corresponding similarity scores
        grid_pts = np.array(list(self.sim_dict[0].keys()))
        sim_scores = np.array(list(self.sim_dict[0].values()))

        if plot_reorient:

            plot_grid = False
            
            # quat = self.init_dict['init_pose'].item()['init_rot_abs']
            # quat = [quat[0], quat[2], quat[1], quat[3]]

            # rot_vec = R.from_quat(quat).as_rotvec(degrees = False)
            # rot_vec[2] = rot_vec[2] + (np.pi/2)
            # rot_mat = R.from_rotvec(rot_vec).as_matrix()[:2, :2]

            # grid_pts = grid_pts @ rot_mat

            grid_pts = self.rot_pts(grid_pts)
        
        score_min, score_max = sim_scores.min(), sim_scores.max()
        score_lims = (score_min, score_max)


        x_min, y_min = grid_pts.min(axis = 0) - self.grid_size
        x_max, y_max = grid_pts.max(axis = 0) + 2 * self.grid_size
        x_lims = (x_min, x_max)
        y_lims = (y_min, y_max)

        #Plotting Grid with Similarity Scores
        if plot_grid:

            eu.plot_grid_lims(ax, 
                        x_lims = x_lims,
                        y_lims = y_lims,
                        grid_size = self.grid_size,
                        plot_all_centers = False)
        
        else:

            ax.set_xlim(x_lims[0], x_lims[1])
            ax.set_ylim(y_lims[0], y_lims[1])
        
        # Define a colormap
        cmap = cm.coolwarm_r
        norm = colors.Normalize(vmin = score_lims[0], vmax = score_lims[1])
        
        score_colors = cmap(norm(sim_scores))
        score_colors = np.array(score_colors)#.squeeze(axis = 1)
        
        # Add colorbar (legend for colormap)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation = "vertical")
        cbar.set_label("Embedding Similarity", labelpad=10)  # Label for the color legend
        cbar.ax.yaxis.set_label_position('right')

        # cbar.ax.set_yticks([val for val in np.linspace(score_lims[0], score_lims[1], 6)])
        # cbar.ax.set_yticklabels([np.round(val, 2) for val in np.linspace(score_lims[0], score_lims[1], 5)])

        cbar.set_ticks([val for val in np.linspace(score_lims[0], score_lims[1], 5)])
        cbar.set_ticklabels([np.round(val, 2) for val in np.linspace(score_lims[0], score_lims[1], 5)])

        #Plot grid point and corresponding color
        ax.scatter(grid_pts[:, 0], grid_pts[:, 1], color = score_colors, s = point_size)
        
        ax.set_title(f'Embedding Similarity Grid')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])

    def rot_pts(self, pts):

        quat = self.init_dict['init_pose'].item()['init_rot_abs']
        quat = [quat[0], quat[2], quat[1], quat[3]]

        rot_vec = R.from_quat(quat).as_rotvec(degrees = False)
        #rot_vec[2] = rot_vec[2] + (np.pi/2)
        rot_mat = R.from_rotvec(rot_vec).as_matrix()[:2, :2]

        pts = pts @ rot_mat

        return pts



class Sim_Grid_Online:

    def __init__(self, 
                 model_name: str = "BLIP2", use_model = None,
                 grid_size: int = 20, 
                 prompt_mode: str = "text", process_sim_mode: str = "mean",
                 scrape_imgs: bool = False, scrape_num: int = 3, scrape_data_dir: str = "data/scraped_imgs/ovon_15",
                 top_n_patch_embeds: int = 1
                 ):

        assert model_name in ["BLIP2", "SED", "clip", "siglip"], "Model name should be : ['BLIP2', 'SED', 'clip', 'siglip']"                
        
        self.model_name = model_name
        self.grid_size = grid_size

        self.prompt_mode = prompt_mode
        self.process_sim_mode = process_sim_mode
        self.scrape_imgs = scrape_imgs
        self.scrape_num = scrape_num
        self.scrape_data_dir = scrape_data_dir
        self.top_n_patch_embeds = top_n_patch_embeds

        if use_model is not None:
            self.model = use_model
            print(f"Using already loaded {self.model_name} model...")
        elif model_name == "BLIP2":
            print(f"Loading {self.model_name} model...")
            self.model = BLIP2_Embed()
        elif model_name in ['clip', 'siglip']:
            print(f"Loading {self.model_name} model...")
            self.model = CLIP_UnifiedEmbedder(backend = model_name)
        else:
            print(f"Loading {self.model_name} model...")
            self.model = OpenClip_Embed()
            


        self.sim_dict = {}
        self.text, self.imgs = None, None
        self.ref_embeds = {} 
        self.ref_imgs = {}

    def load_sim_dict(self, embed_dict,
                      text: str = None,
                      cat_descr: Dict = None):
        r"""Loads similarity dictionary with grid points and corresponding cosine similarity scores. These 
        are calculated in three modes: 
            - image: Uses only image prompt. Can be given as input, or can be retrieved from internet using scrape_imgs arg.
            - text: Uses only text prompt. 
            - multi: Uses both text and image embeds, and combines them in certain ways to produce the scores.        
        """

        print(f"\n Prompt Mode: {self.prompt_mode}\n Process Sim Mode: {self.process_sim_mode}\n Scrape Num: {self.scrape_num}\n")

        assert embed_dict is not None, "Embeddings are not loaded yet! Please run load_embed_init method first."

        if self.scrape_imgs:
            assert text is not None, "Please provide a text prompt to scrape for images."

            #Get Query
            if cat_descr is not None:
                main_cat = text.split(",")[0]
                sub_cat = cat_descr[main_cat][text]

                query = sub_cat
                search_dir = os.path.join(self.scrape_data_dir, main_cat)
            else:
                query = text
                search_dir = self.scrape_data_dir

            #Load images once, and then save them in a ref section
            ref_imgs_key = (query, self.prompt_mode)
            if ref_imgs_key not in self.ref_imgs:
                print(f" Getting new images for prompt: {ref_imgs_key}")                

                imgs = load_images(query = query, num_images = self.scrape_num, save_dir = search_dir)
                assert len(imgs) > 0, "Number of scraped images is zero!"

                self.ref_imgs[ref_imgs_key] = imgs

            # self.imgs = imgs
            self.imgs = self.ref_imgs[ref_imgs_key]


        self.text = text
        self.prompt_mode = self.prompt_mode

        if self.prompt_mode == "text": self.imgs = None
        # elif self.prompt_mode == "image": self.text = None
        elif self.prompt_mode == "multi":
            assert (self.imgs is not None) and (self.text is not None)
        else:
            assert self.prompt_mode in ["text", "image", "multi"], "Please provide a valid input for prompt mode: image, text, multi"

        #Call embeds once, and then save it for reference for further calls
        # multi_target_embeds = self.get_embed_eval(top_n_patch_embeds=self.top_n_patch_embeds) 
        embed_key = (self.text, self.prompt_mode)
        if embed_key not in self.ref_embeds.keys():
            print(f" Getting new embeddings for prompt: {embed_key}")
            self.ref_embeds = {}
            self.ref_embeds[embed_key] = self.get_embed_eval(top_n_patch_embeds=self.top_n_patch_embeds)

        multi_target_embeds = self.ref_embeds[embed_key]
        

        # self.sim_dict = {px : cosine_sim(np.array(target_embed), np.array(px_embed)) for px, px_embed in embed_dict.items()}

        self.sim_dict = {}
        for input_ind in range(multi_target_embeds.shape[0]):
            target_embed = multi_target_embeds[input_ind]
            self.sim_dict[input_ind] = {px : cosine_sim(np.array(target_embed), np.array(px_embed).squeeze(0)) for px, px_embed in embed_dict.items()}

        if self.process_sim_mode is not None and len(self.sim_dict) > 1:
            self.process_sim_dict(process_mode = self.process_sim_mode)
        
    def get_embed_eval(self, top_n_patch_embeds = 1):

        #If text embed is required
        if not (self.prompt_mode == "image"):
            multi_embeds = self.model.get_embed(txt = self.text).cpu().numpy()
        
        #If image embed is required
        if not (self.prompt_mode == "text"):
            
            for img_count, img in enumerate(self.imgs):
                
                if img_count == 0 and self.prompt_mode == "image":

                    if self.model_name == "SED":
                        multi_embeds = self.model.get_embed(image = img, txt = self.text, top_n_patch_embeds = top_n_patch_embeds).cpu().numpy()
                    else:
                        multi_embeds = self.model.get_embed(image = img).cpu().numpy()
                    continue

                    # multi_embeds = self.model.get_embed(image = img, 
                    #                                     txt = self.text if self.model_name == "SED" else None).cpu().numpy()
                    # continue

                if self.model_name == "SED":
                    img_embed = self.model.get_embed(image = img, txt = self.text, top_n_patch_embeds = top_n_patch_embeds).cpu().numpy()
                else:
                    img_embed = self.model.get_embed(image = img).cpu().numpy()
                # img_embed = self.model.get_embed(image = img,
                #                                  txt = self.text if self.model_name == "SED" else None).cpu().numpy()     #Of shape (1, 768)
                multi_embeds = np.concatenate((multi_embeds, img_embed), axis = 0) #Of Shape (num_imgs (+ 1), 768)

        return multi_embeds

    def process_sim_dict(self, process_mode):
        
        assert process_mode in ["mean", "harmonic_mean", "multi_mean"]

        #Gather all the values corresponding to each grid pixel point
        px_pts = self.sim_dict[next(iter(self.sim_dict))].keys()
        values_dict = {px : np.array([sub_sim_dict[px] for sub_sim_dict in self.sim_dict.values()]) for px in px_pts}
        # print(f"PX value shape: {values_dict[next(iter(values_dict))].shape}, {values_dict[next(iter(values_dict))]}")
        
        if process_mode == "mean":
            print(f" Processing Similarity Dictionary: Arithmetic Mean")
            mean_dict = {px : np.mean(vals) for px, vals in values_dict.items()}
            self.sim_dict = {0 : mean_dict}
        
        if process_mode == "harmonic_mean":
            print(" Processing Similarity Dictionary: Harmonic Mean")
            harm_mean_dict = {px : len(vals) / np.sum(1.0 / vals) for px, vals in values_dict.items()}
            self.sim_dict = {0 : harm_mean_dict}

        if process_mode == "multi_mean": #Harmonic_Mean(Text, Mean(Images))
            print(" Processing Similarity Dictionary: Multi Mean")
            multi_mean_dict  = {px : np.array([arr[0], np.mean(arr[1:])]) for px, arr in values_dict.items()}
            # print(f"MEAN PX value shape: {multi_mean_dict[next(iter(multi_mean_dict))].shape}, {multi_mean_dict[next(iter(multi_mean_dict))]}")
            multi_mean_dict = {px : len(vals) / np.sum(1.0 / vals) for px, vals in multi_mean_dict.items()}
            self.sim_dict = {0 : multi_mean_dict}

    def top_k_sims(self, k: int, return_sims: bool = False):
        r"""
        Returns top k pixel positions and their similarity scores
        """

        top_px, top_sims = {}, {}
        for input_ind in self.sim_dict.keys():
            top_items = heapq.nlargest(n = k, 
                                iterable = self.sim_dict[input_ind].items(), 
                                key = lambda item: item[1])
            
            top_px[input_ind] = np.array([list(item[0]) for item in top_items])
            top_sims[input_ind] = np.array([item[1] for item in top_items])

        if not return_sims:  return top_px
        return top_px, top_sims


    # def px_to_hab(self, px_pts):
    #     r"""
    #     Converts from pixels to habitat coordinates
    #     Pipeline:
    #         -> Swap pixel coords (accounting for using cv2 contours)
    #         -> Convert pixels to agent coordinates
    #             -> Inverts y axis 
    #             -> Recenters to pixel origin 
    #             -> Rescales pixels to meters
    #             -> Swaps the x and y positions: (y, x) -> (x, y)
    #         -> Convert agent coords to habitat coords
    #     """

    #     px = px_pts.copy()
    #     px = px[:, ::-1]

    #     #Pixels to Agent Coords
    #     px[:, 0] = self.init_dict['map_shape'][0] - px[:, 0]
        
    #     xy_pts = (px - self.init_dict['pixel_origin']) / self.init_dict['pixels_per_meter']
    #     xy_pts = xy_pts[:, ::-1]

    #     #Agent to Hab Coords
    #     tf_mat_inv = np.linalg.inv(self.tf_hab_to_agent)
    #     xy_pts = np.hstack((xy_pts, np.zeros((xy_pts.shape[0], 1))))
    #     hab_pts = list(map(lambda pt: gu.transform_points(tf_mat_inv, pt[np.newaxis, :]), xy_pts))
    #     hab_pts = np.array(hab_pts).squeeze(1)[:, :2]

    #     return hab_pts
    
    # def hab_to_px(self, hab_pts):
    #     r"""
    #     Converts from habitat coords to pixel coordinates
    #     Pipeline:
    #         -> Convert habitat to agent coordinates
    #         -> Convert agent to pixel coordinates
    #             -> Swap coords
    #             -> Rescales to pixel scale, and recenters to pixel origin
    #             -> Inverts the y axis
    #         -> Swap pixel coords
    #     """

    #     #Habitat to Agent coords
    #     hab = np.hstack((hab_pts, np.zeros((hab_pts.shape[0], 1))))
    #     xy = list(map(lambda pt: gu.transform_points(self.tf_hab_to_agent, pt[np.newaxis, :]), hab))
    #     xy = np.array(xy).squeeze(1)[:, :2]
        
    #     #Agent to Pixel coords
    #     xy = xy[:, ::-1]
    #     px = np.rint(xy * self.init_dict['pixels_per_meter']) + self.init_dict['pixel_origin']
    #     px[:, 0] = self.init_dict['map_shape'][0] - px[:, 0]

    #     return px.astype(int)[:, ::-1]
    
    # def px_to_arr(self, px_pts, arr_origin):
    #     r"""
    #     Returns pixels values accounting for the grid size and array origin
    #     """

    #     px = px_pts.copy()
    #     px = px[:, ::-1]

    #     px[:, 0] = self.init_dict['map_shape'][0] - px[:, 0]
    #     px_rel = (px - self.init_dict['pixel_origin']) / self.init_dict['pixels_per_meter']
    #     px_rel = px_rel[:, ::-1]

    #     arr_pts = px_rel + arr_origin

    #     return arr_pts
    
    # def load_embed_np_arr(self, embed_dict, visualize=False):
    #     r"""
    #     Creates a numpy array with embeddings at relevant positions (zeros at other positions).
    #     This is supposed to represent the map accounting for the grid size.
    #     E.g.
    #         -> Map of size 1000x1000
    #         -> Grid size of 20x20
    #         -> Embedding dimension of 768
    #         Creates an array of shape (50, 50, 768)
    #     """
        
    #     arr_shape = self.init_dict['map_shape'] // self.grid_size
    #     arr_origin = arr_shape // 2
    #     embed_dim = next(iter(embed_dict.values())).shape[0]

    #     embed_arr = np.zeros((arr_shape[0], arr_shape[1], embed_dim))
    #     print(f'Loading array of shape: {embed_arr.shape}')
        
    #     px_to_arr_pos = {px : self.px_to_arr(np.array([px]), arr_origin)[0] \
    #                      for px in embed_dict.keys()}
        
    #     for px_pos in embed_dict.keys():
    #         arr_pos = px_to_arr_pos[px_pos]
    #         embed_arr[int(arr_pos[0]), int(arr_pos[1])] = embed_dict[px_pos]

    #     if visualize:
    #         vis_arr = embed_arr.sum(2)
    #         vis_arr[vis_arr > 0] = 1
    #         plt.imshow(vis_arr)
        
    #     return embed_arr


    def visualize(self, save_dir = None, plot_pts = None, 
                  plot_grid: bool = True, plot_reorient: bool = False,
                  point_size: int = 10,
                  return_fig: bool = True):

        assert len(self.sim_dict) > 0, 'Please load the sim dict first'


        fig, ax = plt.subplots(1, 1, figsize = (2.56, 2.56))
        ax.set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        top_pts = self.top_k_sims(k = 3)[0]
        if plot_reorient: top_pts = self.rot_pts(top_pts)

        ax.scatter(top_pts[:, 0], top_pts[:, 1], 
                        facecolors='white', edgecolors='black', 
                        s=150, linewidths=2.5)
            
        self.plot_scores_in_grid(ax = ax, fig = fig, 
                                    plot_grid = plot_grid,
                                    plot_reorient=plot_reorient,
                                    point_size=point_size)

        if plot_pts is not None:
            if plot_reorient: plot_pts = self.rot_pts(plot_pts)
            ax.scatter(plot_pts[:, 0], plot_pts[:, 1], 
                        facecolors='None', edgecolors='green', 
                        s=150, linewidths=2.5)
            # ax.scatter(plot_pts[:, 0], plot_pts[:, 1], color = 'black')

        
        if return_fig:
            fig.canvas.draw()
            rgb_arr = np.array(fig.canvas.buffer_rgba())[:, :, :3]

            plt.close(fig)
            return rgb_arr
        
        plt.show()

    def plot_img(self, ax: plt.Axes, title: str = 'Image Prompt'):
        ax.imshow(self.img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        
    def plot_scores_in_grid(self, ax: plt.Axes, fig: plt.figure, 
                            plot_grid: bool = False, plot_reorient: bool = False,
                            point_size: int = 10):

        #Get grid points and corresponding similarity scores
        grid_pts = np.array(list(self.sim_dict[0].keys()))
        sim_scores = np.array(list(self.sim_dict[0].values()))

        if plot_reorient:

            plot_grid = False
            
            # quat = self.init_dict['init_pose'].item()['init_rot_abs']
            # quat = [quat[0], quat[2], quat[1], quat[3]]

            # rot_vec = R.from_quat(quat).as_rotvec(degrees = False)
            # rot_vec[2] = rot_vec[2] + (np.pi/2)
            # rot_mat = R.from_rotvec(rot_vec).as_matrix()[:2, :2]

            # grid_pts = grid_pts @ rot_mat

            grid_pts = self.rot_pts(grid_pts)
        
        score_min, score_max = sim_scores.min(), sim_scores.max()
        score_lims = (score_min, score_max)


        x_min, y_min = grid_pts.min(axis = 0) - self.grid_size
        x_max, y_max = grid_pts.max(axis = 0) + 2 * self.grid_size
        x_lims = (x_min, x_max)
        y_lims = (y_min, y_max)

        #Plotting Grid with Similarity Scores
        if plot_grid:

            eu.plot_grid_lims(ax, 
                        x_lims = x_lims,
                        y_lims = y_lims,
                        grid_size = self.grid_size,
                        plot_all_centers = False)
        
        else:

            ax.set_xlim(x_lims[0], x_lims[1])
            ax.set_ylim(y_lims[0], y_lims[1])
        
        # Define a colormap
        cmap = cm.coolwarm_r
        norm = colors.Normalize(vmin = score_lims[0], vmax = score_lims[1])
        
        score_colors = cmap(norm(sim_scores))
        score_colors = np.array(score_colors)#.squeeze(axis = 1)
        
        # # Add colorbar (legend for colormap)
        # cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation = "vertical")
        # cbar.set_label("Embedding Similarity", labelpad=10)  # Label for the color legend
        # cbar.ax.yaxis.set_label_position('right')

        # # cbar.ax.set_yticks([val for val in np.linspace(score_lims[0], score_lims[1], 6)])
        # # cbar.ax.set_yticklabels([np.round(val, 2) for val in np.linspace(score_lims[0], score_lims[1], 5)])

        # cbar.set_ticks([val for val in np.linspace(score_lims[0], score_lims[1], 5)])
        # cbar.set_ticklabels([np.round(val, 2) for val in np.linspace(score_lims[0], score_lims[1], 5)])

        #Plot grid point and corresponding color
        ax.scatter(grid_pts[:, 0], grid_pts[:, 1], color = score_colors, s = point_size)
        
        ax.set_title(f'Embedding Similarity Grid')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])

    def rot_pts(self, pts):

        quat = self.init_dict['init_pose'].item()['init_rot_abs']
        quat = [quat[0], quat[2], quat[1], quat[3]]

        rot_vec = R.from_quat(quat).as_rotvec(degrees = False)
        #rot_vec[2] = rot_vec[2] + (np.pi/2)
        rot_mat = R.from_rotvec(rot_vec).as_matrix()[:2, :2]

        pts = pts @ rot_mat

        return pts
