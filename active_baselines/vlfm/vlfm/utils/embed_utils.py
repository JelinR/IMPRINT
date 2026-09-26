from typing import Any, List, Tuple
from functools import partial
import torch.nn.functional as F

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ast
import os
import re

custom_cmap = ListedColormap(['#edf6f9', '#8d99ae'])
curr_pts_color = '#83c5be'  # '#e29578'
curr_cnts_color = '#83c5be' # '#e29578'
prev_pts_color =  '#e29578' #    '#83c5be'
plt_x_lims = (375, 575)
plt_y_lims = (400, 575)


####################
## Obtaining Grid points inside polygon area
####################

#Obtains all multiples of mult in given range (limits)
def get_multiples_in_range(mult: float, limits: Tuple[float, float]):

    # assert limits[0] < limits[1], 'Please provide limits as (min, max)'
    

    add_factor = limits[0] % mult
    if add_factor != 0:
        add_factor = mult - (limits[0] % mult)
        
    return np.arange(limits[0] + add_factor, limits[1], mult)


def order_pts(points):

    pts = points.copy()
    n_pts = len(points)

    #Obtain centroid and calculate angle to each points
    #Sort by angles to get ordered points
    coord_centroid = lambda coord: sum([pt[coord] for pt in pts]) / n_pts
    angle_btw_pts = lambda pt_1, pt_2: np.atan2( (pt_2[1] - pt_1[1]), (pt_2[0] - pt_1[0]) )
    
    pt_centroid = (coord_centroid(0), coord_centroid(1))
    ordered_pts = sorted( pts, key = lambda pt: angle_btw_pts(pt_1 = pt_centroid, pt_2 = pt) )

    return ordered_pts



is_pt_in_area = lambda cnts, pt: any([cv2.pointPolygonTest(cnt, pt, False) > 0 for cnt in cnts])
unsqueeze = lambda arr, axis: np.expand_dims(arr, axis = axis)

#Plot points from list or array of shape (n, 2)
def plot_arr_pts(ax, arr: np.ndarray, mode:str = 'scatter', color:str = 'gray', label:str = None, pt_size:int = 15, alpha:float = 1):

    assert mode in ['scatter', 'plot'], 'Please provide valid mode: ["scatter", "plot"]'

    if mode == 'scatter':
        ax.scatter([pt[0] for pt in arr], [pt[1] for pt in arr], c = color, label = label, s = pt_size, alpha = alpha)
    else:
        ax.plot([pt[0] for pt in arr], [pt[1] for pt in arr], c = color, label = label, linestyle = '--', linewidth = 1, alpha = alpha)

def square_pts_from_diag(diag_pts: np.ndarray):

    return np.array([
        [diag_pts[0][0], diag_pts[0][1]],
        [diag_pts[1][0], diag_pts[0][1]],
        [diag_pts[1][0], diag_pts[1][1]],
        [diag_pts[0][0], diag_pts[1][1]],
        [diag_pts[0][0], diag_pts[0][1]]
    ])

def plot_default(ax):
        ax.set(xlim = plt_x_lims, ylim = plt_y_lims, yticklabels = [], xticklabels = [])
        ax.tick_params(bottom = False, left = False)


def valid_grid_pts_from_fov(fov_arr: np.ndarray, grid_size: float, show_fig = False):

    #Find Contour Points
    cnts, _ = cv2.findContours(fov_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Squeezes sub-contours along the first axis: (7, 1, 2) -> (7, 2)
    cnts = list(map(lambda c: np.squeeze(c, axis = 1), cnts))
    
    #Filters out sub-contours with less than 3 contours (not a closed shape)
    cnts = [sub_cnt for sub_cnt in cnts if sub_cnt.shape[0] > 2 ]
    if len(cnts) == 0: return None, None

    #Find FOV Area Limits : x_lims, y_lims

    #Get min limits for each sub-contours, and then get the global min limits
    #Similarily for the max limits
    sub_cnts_min = [sub_cnt.min(axis = 0) for sub_cnt in cnts]
    x_min, y_min = np.array(sub_cnts_min).min(axis = 0)

    sub_cnts_max = [sub_cnt.max(axis = 0) for sub_cnt in cnts]
    x_max, y_max = np.array(sub_cnts_max).max(axis = 0)


    #Potential Grid points within Area Boundaries (not FOV, but the square bounding the FOV)
    grid_pts_x = get_multiples_in_range(mult = grid_size, limits = (x_min, x_max))
    grid_pts_y = get_multiples_in_range(mult = grid_size, limits = (y_min, y_max))
    grid_pts = np.array([[x, y] for x in grid_pts_x for y in grid_pts_y]).astype(float)


    #Obtain Valid grid points (lying within FOV) using mask
    valid_mask = list(map(lambda pt: is_pt_in_area(cnts=cnts, pt=pt), grid_pts))
    # valid_mask = valid_mask.astype(bool)
    valid_grid_pts = grid_pts[valid_mask]

    if show_fig:

        fig, ax = plt.subplots(2, 2, figsize = (8, 8))
        fig.subplots_adjust(right = 0.9)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 10

        custom_cmap = ListedColormap(['#edf6f9', '#8d99ae'])

        #Plot Contour Points
        ax[0, 0].set_title('Step 1: \n Contour Points of valid FOV')
        ax[0, 0].imshow(fov_arr, cmap = custom_cmap, alpha = 0.5)
        for cnt in cnts:
            plot_arr_pts(ax = ax[0, 0], arr = cnt, color = '#2b2d42', label = 'Contour Points', alpha = 0.5, pt_size = 5)
        plot_default(ax = ax[0, 0])

        #Plot Contour Points + Area Limits
        ax[0, 1].set_title('Step 2:\n Circumscribed Square Limits')
        ax[0, 1].imshow(fov_arr, cmap = custom_cmap, alpha = 0.5)
        for cnt in cnts:
            plot_arr_pts(ax = ax[0, 1], arr = cnt, color = '#2b2d42', label = 'Contour Points', alpha = 0.5, pt_size = 5)

        limit_pts = np.concatenate((unsqueeze([x_min, y_min], 0), unsqueeze([x_max, y_max], 0)))
        plot_arr_pts(ax = ax[0, 1], arr = limit_pts, color = 'grey', label = 'Area Limits', pt_size = 20)

        square_limit_pts = square_pts_from_diag(diag_pts=limit_pts)
        plot_arr_pts(ax = ax[0, 1], arr = square_limit_pts, mode = 'plot', color = 'grey', label = 'Valid Area Boundary')
        plot_default(ax = ax[0, 1])

        
        #Plot Contour Points + Area Limits + Potential Grid Points
        ax[1, 0].set_title('Step 3:\n Potential Grid Points in Limits')
        ax[1, 0].imshow(fov_arr, cmap = custom_cmap, alpha = 0.5)
        for cnt in cnts:
            plot_arr_pts(ax = ax[1, 0], arr = cnt, color = '#2b2d42', label = 'Contour Points', alpha = 0.5, pt_size = 5)

        limit_pts = np.concatenate((unsqueeze([x_min, y_min], 0), unsqueeze([x_max, y_max], 0)))
        plot_arr_pts(ax = ax[1, 0], arr = limit_pts, color = 'grey', label = 'Area Limits', pt_size = 20)

        square_limit_pts = square_pts_from_diag(diag_pts=limit_pts)
        plot_arr_pts(ax = ax[1, 0], arr = square_limit_pts, mode = 'plot', color = 'grey', label = 'Valid Area Boundary')

        plot_arr_pts(ax = ax[1, 0], arr = grid_pts, color = '#e76f51', label = 'Potential Grid Points')

        plot_default(ax = ax[1, 0])

        #Plot Contour Points + Area Limits + Valid and Invalid Grid Points
        ax[1, 1].set_title('Step 4:\n Points in Contours are valid')
        ax[1, 1].imshow(fov_arr, cmap = custom_cmap, alpha = 0.5)
        for cnt in cnts:
            plot_arr_pts(ax = ax[1, 1], arr = cnt, color = '#2b2d42', label = 'Contour Points', alpha = 0.5, pt_size = 5)

        limit_pts = np.concatenate((unsqueeze([x_min, y_min], 0), unsqueeze([x_max, y_max], 0)))
        plot_arr_pts(ax = ax[1, 1], arr = limit_pts, color = 'grey', label = 'Area Limits', pt_size = 20)

        square_limit_pts = square_pts_from_diag(diag_pts=limit_pts)
        plot_arr_pts(ax = ax[1, 1], arr = square_limit_pts, mode = 'plot', color = 'grey', label = 'Valid Area Boundary')

        plot_arr_pts(ax = ax[1, 1], arr = grid_pts, color = '#e76f51', label = 'Potential Grid Points')
        plot_arr_pts(ax = ax[1, 1], arr = valid_grid_pts, color = '#218380', label = 'Valid Grid Points')  

        plot_default(ax = ax[1, 1])


        handles, labels = ax[1, 1].get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        fig.legend(unique_labels.values(), unique_labels.keys(), loc = 'lower center', ncol=3, bbox_to_anchor = (0.5, 0.0))
    
    return cnts, valid_grid_pts




####################
## Plotting Polygon in Grid Space
####################

def plot_points(ax, pts):

    ax.scatter([pt[0] for pt in pts], [pt[1] for pt in pts], color = curr_pts_color)

def plot_polygon(ax, corners: np.ndarray, order_points = False):

    pts = corners.copy()
    if order_points: pts = order_pts(pts)
    pts = np.concatenate((pts, np.expand_dims(pts[0], axis = 0)), axis = 0)

    ax.plot([pt[0] for pt in pts], [pt[1] for pt in pts], color = curr_cnts_color)

def plot_grid_lims(ax, x_lims, y_lims, grid_size, prev_pts = None, plot_all_centers = True):

    # Generate grid centers
    # x_coords = np.arange(x_lims[0], x_lims[1], grid_size)
    # y_coords = np.arange(y_lims[0], y_lims[1], grid_size)

    x_coords = get_multiples_in_range(mult = grid_size, limits = x_lims)
    y_coords = get_multiples_in_range(mult = grid_size, limits = y_lims)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Plot the grid
    if plot_all_centers: ax.scatter(x_grid, y_grid, color='#d9d9d9', label='Grid Centers', alpha = 0.5)
    
    if prev_pts is not None:
        ax.scatter([pt[0] for pt in prev_pts], [pt[1] for pt in prev_pts], color = prev_pts_color)

    # Draw grid lines
    for x in x_coords:
        ax.axvline(x=x - grid_size/2, color='gray', linestyle='--', linewidth=0.5)
    for y in y_coords:
        ax.axhline(y=y - grid_size / 2, color='gray', linestyle='--', linewidth=0.5)

    # ax.set_xlim(x_lims[0] -grid_size / 2, x_lims[1] - grid_size/2)
    # ax.set_ylim(y_lims[0] -grid_size / 2, y_lims[1] - grid_size/2)
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')


def plot_cnt_in_grid_space(cnts: np.ndarray, valid_grid_pts: np.ndarray, 
                           center_pt: np.ndarray, grid_size: float, 
                           prev_pts = None, show_plot = False,
                           plot_corners = None, gap = 50):
    
    fig, ax = plt.subplots(1, 1, figsize = (2.56, 2.56))

    x_lims = (center_pt[0] - 128, center_pt[0] + 128) 
    y_lims = (center_pt[1] - 128, center_pt[1] + 128)

    #Obtain Previous Valid Grid Points to plot in Grid Space
    if prev_pts is not None:
        if len(prev_pts) < 1:
            prev_pts = None
        else:
            prev_pts = [pt for pt in prev_pts if x_lims[0] < pt[0] < x_lims[1] and y_lims[0] < pt[1]< y_lims[1]]


    #Adjusts figure dimensions
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)


    plot_grid_lims(ax, x_lims = x_lims, y_lims = y_lims, grid_size = grid_size, prev_pts=prev_pts)
    if cnts is not None:
        for cnt in cnts:
            plot_polygon(ax, corners = cnt)

    if valid_grid_pts is not None: 
        plot_points(ax, pts = valid_grid_pts)

    #TODO: Changed for patch
    if plot_corners is not None:
        #Choose specific corners spaced by attribute gap
        if gap > 0:
            plot_inds = np.arange(0, len(plot_corners), gap)
            plot_corners = np.array([plot_corners[ind] for ind in plot_inds])

        rect_colors = get_colors_for_range(num_pts = len(plot_corners))
        rect_colors = rect_colors[::-1]

        for ind, corners in enumerate(plot_corners):
            plot_rect(ax, corners, color = rect_colors[ind])

    if show_plot: plt.show()

    
    #Return as Numpy Array
    fig.canvas.draw()
    rgb_arr = np.array(fig.canvas.buffer_rgba())[:, :, :3]

    plt.close(fig)

    return rgb_arr


def plot_rect(ax, corners, color):

    corners = np.array(corners)
    color = [c / 255 for c in color]

    for pt_ind in range(-1, 3):

        x_pts = corners[[pt_ind, pt_ind + 1]][:, 0]
        y_pts = corners[[pt_ind, pt_ind + 1]][:, 1]

        ax.plot(x_pts, y_pts, color = color, alpha = 1.0)

########################
## Obtaining Corners from Fog of War
########################

#Obtain valid FOV from obstacle map
def get_valid_fov(obs_map):

    valid_fov = np.zeros((*obs_map._map.shape[:2],), dtype=np.uint8)
    # Draw explored area in light green
    valid_fov[obs_map.explored_area == 1] = 255
    # Draw unnavigable areas in gray
    valid_fov[obs_map._navigable_map == 0] = 255
    # Draw obstacles in black
    valid_fov[obs_map._map == 1] = 255

    # valid_fov = cv2.flip(valid_fov, 0)

    return valid_fov

#Perform a closing morphology (dilation and erosion), and perform polygon approximation
def process_fow(fow):

    fow_closing = cv2.morphologyEx(fow, 
                                cv2.MORPH_CLOSE, 
                                kernel = np.ones((7, 7), np.uint8),
                                iterations=1)
        
    return fow_closing

# #Obtain polygon corners using Shi-Tomasi Corner Detection
# def get_polygon_corners(img, top_n_corners = 10):

#     corners = cv2.goodFeaturesToTrack(img, maxCorners=top_n_corners, qualityLevel=0.01, minDistance=2)
#     corner_points = [(int(x), int(y)) for [x, y] in corners.reshape(-1, 2)]

#     return corner_points




########################
## Patch to Map Corners
########################

#Returns next closest number (including the number) divisible by <div>
closest_div_num = lambda num, div: num if (num % div) == 0 else num + ( div - (num % div) )

#Interpolates depth image to next closest divisible dimensions
def interpolate_depth(depth: np.ndarray, size_divisbility:Tuple = (24, 24)):

    assert len(depth.shape) == 2, 'Please provide depth with shape [h, w]'
    if torch.is_tensor(depth):
        depth_tensor = depth.unsqueeze(0).unsqueeze(0)
    else:
        depth_tensor = torch.tensor(depth).unsqueeze(0).unsqueeze(0)

    scaled_shape = (
        closest_div_num(num = depth.shape[0], div = size_divisbility[0]),
        closest_div_num(num = depth.shape[1], div = size_divisbility[1])
    )


    depth_inter = F.interpolate(depth_tensor, scaled_shape, mode='bilinear', align_corners=True)
    return np.array(depth_inter[0][0])

#Returns a binary 2D array of corners of the image patches
def get_patch_img_corners(depth_inter: np.ndarray, size_divisibility: Tuple = (24, 24)):

    """
    depth_inter: Interpolated Depth - (h, w)
    size_divisibility: Number of patches along each dimension
    """
    
    y_lim, x_lim = depth_inter.shape
    patch_size = (y_lim // size_divisibility[0], x_lim // size_divisibility[1])

    x_pts = np.arange(0, x_lim + 1, patch_size[1])
    y_pts = np.arange(0, y_lim + 1, patch_size[0])

    curb_to_range = lambda pts, lim: np.array([pt if pt < lim - 1 else lim - 1 for pt in pts])

    x_pts = curb_to_range(x_pts, x_lim)
    y_pts = curb_to_range(y_pts, y_lim)

    patch_img_corners = np.zeros_like(depth_inter)
    x_mesh, y_mesh = np.meshgrid(x_pts, y_pts)

    patch_img_corners[y_mesh, x_mesh] = 1

    return patch_img_corners

#Color map for a given number of points
def get_colors_for_range(num_pts):
    # cmap = plt.cm.berlin
    cmap = plt.cm.coolwarm
    norm = np.linspace(0, 1, num_pts)

    colors = [cmap(val) for val in norm]
    colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in colors]

    return colors

#Overlay points on image
def impose_pts_on_rgb(rgb: np.ndarray, patch_img_corners: np.ndarray, 
                      size_divisibility: Tuple = (24, 24),
                      show_img: bool = False):

    radius = 1
    color_count = 0
    thickness = 2

    x_pts, y_pts = np.where(patch_img_corners)
    
    # num_pts = (size_divisibility[0] + 1) * (size_divisibility[1] + 1)
    num_pts = int(patch_img_corners.sum())
    colors = get_colors_for_range(num_pts)

    for (x, y) in zip(x_pts, y_pts):

        rgb = cv2.circle(rgb, (x, y), radius, colors[color_count], thickness)
        color_count += 1

    if show_img:
        plt.imshow(rgb)

    return rgb

#Plot rectangles on image
def overlay_rects_in_img(img: np.ndarray, rect_corners: np.ndarray, 
                         gap: int):
    """"
    Inputs:
    - img: (h, w, 3)
    - rect_corners: (num_rects, 4, 2)  
    - alpha_rect: integer  
    """
    img = img.astype(np.uint8)

    #Choose specific corners spaced by attribute gap
    if gap > 0:
        plot_inds = np.arange(0, len(rect_corners), gap)
        plot_corners = np.array([rect_corners[ind] for ind in plot_inds])
    else:
        plot_corners = rect_corners.copy()

    # print(plot_corners)

    #Get corresponding colormap for number of plot_corners
    patch_colors = get_colors_for_range(num_pts=len(plot_corners))

    #Plot each corner with specific 
    for ind, corners_px in enumerate(plot_corners):

        corners_px = np.expand_dims(corners_px, axis = 0)
        cv2.drawContours(img, corners_px, -1, patch_colors[ind], 2)

    return img


#Convert to pixel scale
def conv_to_px(xy_pts, pixels_per_meter = 20, pixel_center = [500, 500], map_lims = [1000, 1000]):
    px_pts = np.rint(xy_pts[:, ::-1] * pixels_per_meter) + pixel_center

    px_pts[:, 0] = map_lims[0] - px_pts[:, 0]

    px_pts = px_pts.astype(int)

    return px_pts

# #Convert to pixel scale
# def conv_to_px_t(xy_pts, pixels_per_meter = 20, pixel_center = [500, 500], 
#                  map_lims = [1000, 1000], device = 'cpu'):

#     #Exchanges (x, y) coords, Changes to pixel scale, and shifts origin to pixel origin
#     px = torch.round(torch.flip(xy_pts, dims=[1]) * pixels_per_meter) + torch.Tensor(pixel_center).to(device)
    
#     #Flips y axis
#     px[:, 0] = map_lims[0] - px[:, 0]

#     return px.int()


#Obtain all grid corners given the top right corner point
def rect_corners(tl_pt, grid_size):
    
    return [
        [tl_pt[0], tl_pt[1]],
        [tl_pt[0], tl_pt[1] + grid_size[1]],
        [tl_pt[0] + grid_size[0], tl_pt[1] + grid_size[1]],
        [tl_pt[0] + grid_size[0], tl_pt[1]]
    ]


#Given Patch (grid cell) number, return the flattened index
def flat_ind_from_grid_ind(grid_ind: Any, n_cols: int):

    assert grid_ind[1] < n_cols, "Column Value should be less than total columns as they range from zero onwards."

    row, col = grid_ind
    return row * n_cols + col

#Given the flattened index, return the patch (grid cell) number
def grid_ind_from_flat_ind(flat_ind: int, n_cols: int):

    grid_inds = ( flat_ind // n_cols, flat_ind % n_cols )
    return grid_inds

#Returns the flattened indices for corners corresponing to a specific patch
def get_patch_corner_flat_inds(patch_num: int, n_patch_cols: int):

    """
    Inputs:
    - patch_num ranges from 0 onwards
    - img_shape: [h, w]
    - grid_size: [h, w]

    Output:
    - flat_pt_inds: List of shape - (4, 2). Four corners of the patch.
    """

    #Top Left grid point indices of the target patch (grid cell)
    # tl_pt_grid_inds = ( patch_num // n_patch_cols, patch_num % n_patch_cols )
    tl_pt_grid_inds = grid_ind_from_flat_ind(patch_num, n_patch_cols)

    #2D Corner Indices of the target patch
    grid_pt_inds = rect_corners(tl_pt_grid_inds, (1, 1))
    
    #Flat Corner Indices of the target patch
    #Number of points is one more than the number of patches along a dimension
    get_flat_ind = partial(flat_ind_from_grid_ind, n_cols = n_patch_cols + 1)
    flat_pt_inds = list(map(get_flat_ind, grid_pt_inds))

    return flat_pt_inds

#Creates a Dictionary for mapping each Patch to its corresponding corners on the Obstacle Map
def map_patch_to_corners(cloud_pts: np.ndarray,
                         n_patch_dims: Tuple,
                         px_per_meter: int, px_center: Tuple, 
                         map_lims: Tuple):
    """
    Inputs:
    - cloud_pts: [num_pts, 3]
    - n_patch_dims: Number of Patches along each dimension
    - px_per_meter: pixels per meter
    - px_center: pixel center on the map
    - map_lims: map dimensions

    Outputs:
    - patch_to_corners: Dict[ Patch_Number -> ( Obstacle Map Pixel Corners, Flattened Patch Corner Indices ) ]
    """

    #Number of grid points along the row and columns (including the boundary points)
    # n_col_pts = img_shape[1]//patch_size[1] + 1
    # n_row_pts = img_shape[0]//patch_size[0] + 1

    # n_patches = (n_col_pts - 1) * (n_row_pts - 1)
    # patch_to_corners = {}

    n_patch_rows, n_patch_cols = n_patch_dims
    n_patches = n_patch_rows * n_patch_cols
    patch_to_corners = {}

    for patch_num in range(n_patches):

        #Flattened indices for the four corners of a given patch
        corner_inds = get_patch_corner_flat_inds(patch_num = patch_num, n_patch_cols = n_patch_cols)

        #Get Pixel locations of the four patch corners
        #These pixel locations belong to the obstacle map
        corner_pts = cloud_pts[corner_inds, :2]
        corner_pxs = conv_to_px(corner_pts, px_per_meter, px_center, map_lims)
        
        #Save the pixel corners and the flattened corner indices
        patch_to_corners[patch_num] = (corner_pxs, corner_inds)

    return patch_to_corners


def corners_of_patch(patch_id: int, n_patches: int, img_size: Tuple):

    patch_size = ( img_size[0]//n_patches, img_size[1]//n_patches )

    tl_corner = grid_ind_from_flat_ind(patch_id, n_cols = n_patches)
    tl_corner = ( tl_corner[0] * patch_size[0], tl_corner[1] * patch_size[1] )

    corners = rect_corners(tl_corner, patch_size)

    return np.array(corners)

#Returns Normalized Intersection area between two contours
def area_patch_over_grid(patch_cnts:np.ndarray, grid_cnts:np.ndarray):
    """
    patch_cnts: (4, 2)
    grid_cnts: (4, 2)
    
    This accounts for cases where the patch is flattened to one dimension or even a point.
    """

    #Change shape to (4, 1, 2)
    patch_cnts = np.expand_dims(patch_cnts, axis = 1)
    grid_cnts = np.expand_dims(grid_cnts, axis = 1)

    #Total area of mapped patch
    patch_area = cv2.contourArea(patch_cnts)

    #If patch is flattened from a 2D rectangle to a line, 
    # remove the redundant contour points. This is useful in getting 
    # the intersection contours
    if patch_area == 0: 
        patch_cnts = np.unique(patch_cnts, axis = 0)

        #If patch is a point, then return 1
        if len(patch_cnts) <= 1: return 1
    

    # Compute the intersection area
    intersect_area, intersect_cnts = cv2.intersectConvexConvex(patch_cnts, grid_cnts)
    
    #If patch is flattened, get the amount of overlap of the line segment
    if patch_area == 0:
        
        #If there is no intersection, return 0
        if intersect_cnts is None: return 0
        if len(intersect_cnts) == 1: return 0

        #Get ratio of overlapping length of line segment
        patch_area = np.linalg.norm(patch_cnts[0] - patch_cnts[1])
        intersect_area = np.linalg.norm(intersect_cnts[0] - intersect_cnts[1])
    
    return intersect_area / patch_area


#######
##Floor Utils
#######

def floor_name_from_num(floor_num:int):

    return f'floor_{"p" if floor_num >= 0 else "n"}{abs(floor_num)}'


def read_valid_scenes(base_path="habitat-lab/data/datasets/objectnav/hm3d/v2", split="val"):
    """
    Reads a text file containing scene data and returns a list of scenes.
    
    Expected format for each line:
    <int>,<file_path>,<file_name>, [<float>, <float>, <float>], [<float>, <float>, <float>, <float>]
    
    For example:
    27,data/scene_datasets/hm3d_v0.2/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb,
    4ok3usBNeis.basis.glb_tv_monitor, [1.27888, 2.06447, 5.79586],[0, 0.29372, 0, 0.95589]
    
    Parameters:
        base_path (str): Base directory for the dataset.
        split (str): Which data split to use (e.g., 'val').
        
    Returns:
        valid_eps (list): A list of dictionaries, each representing a scene.
    """
    valid_eps = []
    
    # Construct the file path to the scenes file.
    # Adjust the file name if necessary.
    file_name = f"valid_{split}_episodes.txt"
    file_path = os.path.join(base_path, split, file_name)
    
    # Regular expression to split on commas not inside square brackets.
    pattern = r',\s*(?![^\[]*\])'
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                    
                # Split the line into parts while preserving list contents.
                fields = re.split(pattern, line)
                
                # Construct a dictionary for this scene.
                scene = {
                    "id": int(fields[0]),
                    "file_path": fields[1],
                    "file_name": fields[2],
                    "coordinates": ast.literal_eval(fields[3]),
                    "quaternion": ast.literal_eval(fields[4])
                }
                valid_eps.append(scene)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return valid_eps