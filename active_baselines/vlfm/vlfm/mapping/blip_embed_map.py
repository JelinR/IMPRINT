from typing import Any, List, Tuple

import numpy as np
import torch
import cv2
from math import ceil


from vlfm.mapping.obstacle_map import ObstacleMap
import vlfm.utils.embed_utils as eu
from functools import partial

try:
    from vlfm.vlm.blip2_embed import BLIP2_Embed
except ModuleNotFoundError:
    print(f"Could not load BLIP2.")

try:
    from vlfm.vlm.clip_embed import CLIP_UnifiedEmbedder
except ModuleNotFoundError:
    print(f"Could not load clip/siglip.")

#Timing update_map
import time
import os
save_times_path = "/mnt/vlfm_query_embed/timings/nav/update_map/text/blip/times_grid_1.txt"
os.makedirs(os.path.dirname(save_times_path), exist_ok=True)



class Obstacle_Embed_BLIP_Map(ObstacleMap):

    def __init__(self,
                    cell_size: float = 20,
                    embed_model_name: str = "BLIP2",
                    *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.embed_model_name = embed_model_name

        if self.embed_model_name == "BLIP2":
            self._embed_model = BLIP2_Embed()
        elif self.embed_model_name in ['clip', 'siglip']:
            self._embed_model = CLIP_UnifiedEmbedder(backend = self.embed_model_name)
        else:
            raise ValueError(f"Invalid embed model name : {self.embed_model_name}! Options: BLIP2, clip, siglip")

        self.curr_cnts = None
        self.curr_grid_pts = None
        self.cell_size = cell_size
        self._embed_dict = {}

        #TODO: Added for embed, floor
        self.curr_floor_name = "floor_0"

    def reset(self):
        super().reset()
        self._embed_dict = {}

    def update_map(self, rgb: np.ndarray, agent_height: float, **kwargs):

        start_time = time.time()

        super().update_map(**kwargs)

        #TODO: Added for floor
        # curr_floor_num = ceil(agent_height) // 2
        # self.curr_floor_name = eu.floor_name_from_num(curr_floor_num)
        
        # print(f"\n Current Floor Name: {self.curr_floor_name}")
        # print(f" Current Floors: {list(self._embed_dict.keys())}")

        is_new_floor = self.curr_floor_name not in self._embed_dict.keys()

        #TODO: Added for floor
        if is_new_floor:
            print(f"Entering New Floor")
            self._embed_dict[ self.curr_floor_name ] = {}          

        #Applies Closing Morphological operation to the current FOV
        processed_fov = eu.process_fow(self._embed_map)

        #Obtains all contours for the FOV, and the current valid grid cell centers
        self.curr_cnts, self.curr_grid_pts = eu.valid_grid_pts_from_fov(fov_arr = processed_fov,
                                                          grid_size = self.cell_size,
                                                          show_fig = False)
        
        if self.curr_grid_pts is not None:
            img_embed = self._embed_model.get_embed(image = rgb).cpu().numpy()
            print('BLIP Embedding Shape: ', img_embed.shape)

            #For each grid point, update the grid embedding if it already exists, 
            # or add a new entry to the dictionary
            for grid_pt in self.curr_grid_pts:

                if len(self._embed_dict[self.curr_floor_name].keys()) > 0:
                    if tuple(grid_pt) in self._embed_dict[self.curr_floor_name].keys():

                        prev_grid_embed = self._embed_dict[self.curr_floor_name][tuple(grid_pt)].copy()
                        new_grid_embed = (img_embed + prev_grid_embed) / 2
                        self._embed_dict[self.curr_floor_name][tuple(grid_pt)] = new_grid_embed
                    else:
                        self._embed_dict[self.curr_floor_name][tuple(grid_pt)] = img_embed
                else:
                    self._embed_dict[self.curr_floor_name][tuple(grid_pt)] = img_embed


        time_taken = time.time() - start_time
        with open(save_times_path, 'a') as file:
            file.write(str(time_taken) + "\n")
    

    def visualize_embed(self, center_pt: np.ndarray):

        center_px = eu.conv_to_px(xy_pts = center_pt[np.newaxis, :],
                                    pixels_per_meter=self.pixels_per_meter,
                                    pixel_center = self._episode_pixel_origin,
                                    map_lims = self._map.shape)

        return eu.plot_cnt_in_grid_space(
            cnts = self.curr_cnts,
            valid_grid_pts = self.curr_grid_pts,
            center_pt = center_px.squeeze(axis = 0),
            grid_size = self.cell_size,
            prev_pts = list(self._embed_dict[self.curr_floor_name].keys()),
            show_plot = False,
            plot_corners = None
        )



