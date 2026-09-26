from typing import Any, List, Tuple

import numpy as np
import torch
import cv2
from math import ceil


from vlfm.mapping.obstacle_map import ObstacleMap
import vlfm.utils.embed_utils as eu
from functools import partial

from vlfm.vlm.sed import SED_Model

class Obstacle_Embed_SED_Map(ObstacleMap):

    def __init__(self,
                    cell_size: float = 20,
                    size_div: Tuple = (24, 24),
                    patch_min_height: float = 0.0,
                    patch_max_height: float = 5.0,
                    valid_corner_thresh: int = 3.0,
                    patch_over_grid_thresh: float = 0.25,
                    visualize_patch: bool = False,
                    vis_patch_step: int = 50,
                    *args, **kwargs):


        super().__init__(*args, **kwargs)

        self.embed_model_name = "SED"
        self._embed_model = SED_Model()

        self.curr_cnts = None
        self.curr_grid_pts = None
        self.cell_size = cell_size
        self._embed_dict = {}
        self.visualize_patch = visualize_patch
        self.vis_patch_step = vis_patch_step
        self.patch_over_grid_thresh = patch_over_grid_thresh

        self.size_div = size_div
        self.patch_min_height = patch_min_height
        self.patch_max_height = patch_max_height 
        self.valid_corner_thresh = valid_corner_thresh 
        self.valid_patch_to_corners = {}      

        #TODO: Added for embed, floor
        self.curr_floor_name = None
        
    def reset(self):
        super().reset()
        self._embed_dict = {}

    def update_map(self, rgb: np.ndarray, agent_height: float, **kwargs):

        super().update_map(**kwargs)

        #TODO: Added for floor
        curr_floor_num = ceil(agent_height) // 2
        self.curr_floor_name = eu.floor_name_from_num(curr_floor_num)
        
        print(f"\n Current Floor Name: {self.curr_floor_name}")
        print(f"Current Floors: {list(self._embed_dict.keys())}")

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

            #So at this point, we have the current valid grid centers,
            # and the patch corners' locations. We now try to map each 
            # valid grid center to patch indices inside the grid cell

            #Maps each valid grid center to patch indices inside the grid
            grid_pt_to_patch_inds = {}
        
            for grid_pt in self.curr_grid_pts:
                
                #Obtain Grid Cell Corners
                tl_grid_corner = ( grid_pt[0] - self.cell_size/2, grid_pt[1] - self.cell_size/2 )
                grid_cnts = eu.rect_corners(tl_pt = tl_grid_corner, grid_size = (self.cell_size, self.cell_size))
                grid_cnts = np.array(grid_cnts).astype(np.float32)
                
                # Patches that share at least <threshold> of their area with the grid cell 
                # are considered present inside the grid cell
                is_patch_present = {k : eu.area_patch_over_grid(patch_cnts = v[0], grid_cnts = grid_cnts) >= self.patch_over_grid_thresh \
                            for (k, v) in self.valid_patch_to_corners.items()}


                # # Checks if all patch corner points are inside the Grid Cell
                # grid_cnts = np.expand_dims(grid_cnts, axis = 0)
                # is_present = {k: False for k in self.valid_patch_to_corners.keys()}
                # for (patch_flat_ind, (corner_pts, corner_inds)) in self.valid_patch_to_corners.items():
                    
                #     corners_in_grid = [eu.is_pt_in_area(cnts = grid_cnts, pt = np.array(pt).astype(float)) for pt in corner_pts]
                #     if sum(corners_in_grid) >= 0: 
                #         is_present[patch_flat_ind] = True 

                #Assigns present patches to the grid cell
                patches_in_grid = [k for k in is_patch_present.keys() if is_patch_present[k]]

                if len(patches_in_grid) > 0:
                    grid_pt_to_patch_inds[tuple(grid_pt)] = patches_in_grid

            #Obtain SED embedding for the img
            # img_embed = np.random.random((1, 768, 24, 24))
            img_embed = self._embed_model.get_patch_embeds(img_rgb = rgb)
            img_embed = img_embed.cpu().numpy()
                
            print('SED Embedding Shape: ', img_embed.shape)
            

            #For each Grid Cell, save a corresponding grid embedding
            for grid_pt, grid_patches in grid_pt_to_patch_inds.items():

                #List of all present patches' embeddings
                patch_embeds = list(map(lambda patch_ind: self.get_embed_at(img_embed, patch_ind), grid_patches))
                patch_embeds = np.array(patch_embeds)
                
                #Get the mean of present patch embeddings
                grid_embed = patch_embeds.mean(axis = 0)

                #If grid point is already present, then take the mean with previous embedding
                #Else save the grid point and the grid_embed
                if len(self._embed_dict[self.curr_floor_name].keys()) > 0:
                    if grid_pt in self._embed_dict[self.curr_floor_name].keys():

                        prev_grid_embed = self._embed_dict[self.curr_floor_name][tuple(grid_pt)].copy()
                        new_grid_embed = (grid_embed + prev_grid_embed) / 2

                        self._embed_dict[self.curr_floor_name][tuple(grid_pt)] = new_grid_embed
                    else:
                        self._embed_dict[self.curr_floor_name][tuple(grid_pt)] = grid_embed
                else:
                    self._embed_dict[self.curr_floor_name][tuple(grid_pt)] = grid_embed

    
    def get_embed_at(self, embed: np.ndarray, patch_num: int):

        n_patch_cols = embed.shape[-1]
        grid_ind = eu.grid_ind_from_flat_ind(flat_ind = patch_num, n_cols = n_patch_cols)

        return embed[0, :, grid_ind[0], grid_ind[1]]
    
    
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



