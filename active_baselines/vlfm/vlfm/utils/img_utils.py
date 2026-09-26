# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List, Tuple, Union

import cv2
import numpy as np

#Rotates an image counter-clockwise
def rotate_image(
    image: np.ndarray,
    radians: float,
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> np.ndarray:
    """Rotate an image by the specified angle in radians.

    Args:
        image (numpy.ndarray): The input image.
        radians (float): The angle of rotation in radians.

    Returns:
        numpy.ndarray: The rotated image.
    """
    height, width = image.shape[0], image.shape[1]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(radians), 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=border_value)

    return rotated_image


def place_img_in_img(img1: np.ndarray, img2: np.ndarray, row: int, col: int) -> np.ndarray:
    """Place img2 in img1 such that img2's center is at the specified coordinates (xy)
    in img1.

    Args:
        img1 (numpy.ndarray): The base image.
        img2 (numpy.ndarray): The image to be placed.


    Returns:
        numpy.ndarray: The updated base image with img2 placed.
    """
    assert 0 <= row < img1.shape[0] and 0 <= col < img1.shape[1], "Pixel location is outside the image."

    #Remember that we are dealing with numpy arrays, and so the coord system is (x positive right, y positive down)
    #Row corresponds to going along the y axis (down), and col corresponds to going along the x axis (right)
    #We get the corners' locations using (row, col) and img2 shape, that is, we get the area where we need to place the img2 on img1
    top = row - img2.shape[0] // 2
    left = col - img2.shape[1] // 2
    bottom = top + img2.shape[0]
    right = left + img2.shape[1]

    #Gets the area in img1 to be edited, where img2 needs to added onto
    #Curbs the calculated corners within img1's corners
    img1_top = max(0, top)
    img1_left = max(0, left)
    img1_bottom = min(img1.shape[0], bottom)
    img1_right = min(img1.shape[1], right)

    #Crops img2 to make it fit within img1's selected area
    img2_top = max(0, -top)
    img2_left = max(0, -left)
    img2_bottom = img2_top + (img1_bottom - img1_top)
    img2_right = img2_left + (img1_right - img1_left)

    img1[img1_top:img1_bottom, img1_left:img1_right] = img2[img2_top:img2_bottom, img2_left:img2_right]

    return img1


def monochannel_to_inferno_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a monochannel float32 image to an RGB representation using the Inferno
    colormap.

    Args:
        image (numpy.ndarray): The input monochannel float32 image.

    Returns:
        numpy.ndarray: The RGB image with Inferno colormap.
    """
    # Normalize the input image to the range [0, 1]
    min_val, max_val = np.min(image), np.max(image)
    peak_to_peak = max_val - min_val
    if peak_to_peak == 0:
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / peak_to_peak

    # Apply the Inferno colormap
    inferno_colormap = cv2.applyColorMap((normalized_image * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    return inferno_colormap


def resize_images(images: List[np.ndarray], match_dimension: str = "height", use_max: bool = True) -> List[np.ndarray]:
    """
    Resize images to match either their heights or their widths.

    Args:
        images (List[np.ndarray]): List of NumPy images.
        match_dimension (str): Specify 'height' to match heights, or 'width' to match
            widths.

    Returns:
        List[np.ndarray]: List of resized images.
    """
    if len(images) == 1:
        return images

    if match_dimension == "height":
        if use_max:
            new_height = max(img.shape[0] for img in images)
        else:
            new_height = min(img.shape[0] for img in images)
        resized_images = [
            cv2.resize(img, (int(img.shape[1] * new_height / img.shape[0]), new_height)) for img in images
        ]
    elif match_dimension == "width":
        if use_max:
            new_width = max(img.shape[1] for img in images)
        else:
            new_width = min(img.shape[1] for img in images)
        resized_images = [cv2.resize(img, (new_width, int(img.shape[0] * new_width / img.shape[1]))) for img in images]
    else:
        raise ValueError("Invalid 'match_dimension' argument. Use 'height' or 'width'.")

    return resized_images


def crop_white_border(image: np.ndarray) -> np.ndarray:
    """Crop the image to the bounding box of non-white pixels.

    Args:
        image (np.ndarray): The input image (BGR format).

    Returns:
        np.ndarray: The cropped image. If the image is entirely white, the original
            image is returned.
    """
    # Convert the image to grayscale for easier processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the bounding box of non-white pixels
    non_white_pixels = np.argwhere(gray_image != 255)

    if len(non_white_pixels) == 0:
        return image  # Return the original image if it's entirely white

    min_row, min_col = np.min(non_white_pixels, axis=0)
    max_row, max_col = np.max(non_white_pixels, axis=0)

    # Crop the image to the bounding box
    cropped_image = image[min_row : max_row + 1, min_col : max_col + 1, :]

    return cropped_image


def pad_to_square(
    img: np.ndarray,
    padding_color: Tuple[int, int, int] = (255, 255, 255),
    extra_pad: int = 0,
) -> np.ndarray:
    """
    Pad an image to make it square by adding padding to the left and right sides
    if its height is larger than its width, or adding padding to the top and bottom
    if its width is larger.

    Args:
        img (numpy.ndarray): The input image.
        padding_color (Tuple[int, int, int], optional): The padding color in (R, G, B)
            format. Defaults to (255, 255, 255).

    Returns:
        numpy.ndarray: The squared and padded image.
    """
    height, width, _ = img.shape
    larger_side = max(height, width)
    square_size = larger_side + extra_pad
    padded_img = np.ones((square_size, square_size, 3), dtype=np.uint8) * np.array(padding_color, dtype=np.uint8)
    padded_img = place_img_in_img(padded_img, img, square_size // 2, square_size // 2)

    return padded_img


def pad_larger_dim(image: np.ndarray, target_dimension: int) -> np.ndarray:
    """Pads an image to the specified target dimension by adding whitespace borders.

    Args:
        image (np.ndarray): The input image as a NumPy array with shape (height, width,
            channels).
        target_dimension (int): The desired target dimension for the larger dimension
            (height or width).

    Returns:
        np.ndarray: The padded image as a NumPy array with shape (new_height, new_width,
            channels).
    """
    height, width, _ = image.shape
    larger_dimension = max(height, width)

    if larger_dimension < target_dimension:
        pad_amount = target_dimension - larger_dimension
        first_pad_amount = pad_amount // 2
        second_pad_amount = pad_amount - first_pad_amount

        if height > width:
            top_pad = np.ones((first_pad_amount, width, 3), dtype=np.uint8) * 255
            bottom_pad = np.ones((second_pad_amount, width, 3), dtype=np.uint8) * 255
            padded_image = np.vstack((top_pad, image, bottom_pad))
        else:
            left_pad = np.ones((height, first_pad_amount, 3), dtype=np.uint8) * 255
            right_pad = np.ones((height, second_pad_amount, 3), dtype=np.uint8) * 255
            padded_image = np.hstack((left_pad, image, right_pad))
    else:
        padded_image = image

    return padded_image


def pixel_value_within_radius(
    image: np.ndarray,
    pixel_location: Tuple[int, int],
    radius: int,
    reduction: str = "median",
) -> Union[float, int]:
    """Returns the maximum pixel value within a given radius of a specified pixel
    location in the given image.

    Args:
        image (np.ndarray): The input image as a 2D numpy array.
        pixel_location (Tuple[int, int]): The location of the pixel as a tuple (row,
            column).
        radius (int): The radius within which to find the maximum pixel value.
        reduction (str, optional): The method to use to reduce the cropped image to a
            single value. Defaults to "median".

    Returns:
        Union[float, int]: The maximum pixel value within the given radius of the pixel
            location.
    """
    # Ensure that the pixel location is within the image
    assert (
        0 <= pixel_location[0] < image.shape[0] and 0 <= pixel_location[1] < image.shape[1]
    ), "Pixel location is outside the image."

    top_left_x = max(0, pixel_location[0] - radius)
    top_left_y = max(0, pixel_location[1] - radius)
    bottom_right_x = min(image.shape[0], pixel_location[0] + radius + 1)
    bottom_right_y = min(image.shape[1], pixel_location[1] + radius + 1)
    cropped_image = image[top_left_x:bottom_right_x, top_left_y:bottom_right_y]

    # Draw a circular mask for the cropped image
    circle_mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
    circle_mask = cv2.circle(
        circle_mask,
        (radius, radius),
        radius,
        color=255,
        thickness=-1,
    )
    overlap_values = cropped_image[circle_mask > 0]
    # Filter out any values that are 0 (i.e. pixels that weren't seen yet)
    overlap_values = overlap_values[overlap_values > 0]
    if overlap_values.size == 0:
        return -1
    elif reduction == "mean":
        return np.mean(overlap_values)  # type: ignore
    elif reduction == "max":
        return np.max(overlap_values)
    elif reduction == "median":
        return np.median(overlap_values)  # type: ignore
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")


def median_blur_normalized_depth_image(depth_image: np.ndarray, ksize: int) -> np.ndarray:
    """Applies a median blur to a normalized depth image.

    This function first converts the normalized depth image to a uint8 image,
    then applies a median blur, and finally converts the blurred image back
    to a normalized float32 image.

    Args:
        depth_image (np.ndarray): The input depth image. This should be a
            normalized float32 image.
        ksize (int): The size of the kernel to be used in the median blur.
            This should be an odd number greater than 1.

    Returns:
        np.ndarray: The blurred depth image. This is a normalized float32 image.
    """
    # Convert the normalized depth image to a uint8 image
    depth_image_uint8 = (depth_image * 255).astype(np.uint8)

    # Apply median blur
    blurred_depth_image_uint8 = cv2.medianBlur(depth_image_uint8, ksize)

    # Convert the blurred image back to a normalized float32 image
    blurred_depth_image = blurred_depth_image_uint8.astype(np.float32) / 255

    return blurred_depth_image


def reorient_rescale_map(vis_map_img: np.ndarray) -> np.ndarray:
    """Reorient and rescale a visual map image for display.

    This function preprocesses a visual map image by:
    1. Cropping whitespace borders
    2. Padding the smaller dimension to at least 150px
    3. Padding the image to a square
    4. Adding a 50px whitespace border

    Args:
        vis_map_img (np.ndarray): The input visual map image

    Returns:
        np.ndarray: The reoriented and rescaled visual map image
    """
    # Remove unnecessary white space around the edges
    vis_map_img = crop_white_border(vis_map_img)
    # Make the image at least 150 pixels tall or wide
    vis_map_img = pad_larger_dim(vis_map_img, 150)
    # Pad the shorter dimension to be the same size as the longer
    vis_map_img = pad_to_square(vis_map_img, extra_pad=50)
    # Pad the image border with some white space
    vis_map_img = cv2.copyMakeBorder(vis_map_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return vis_map_img


def remove_small_blobs(image: np.ndarray, min_area: int) -> np.ndarray:
    # Find all contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate area of the contour
        area = cv2.contourArea(contour)

        # If area is smaller than the threshold, remove the contour
        if area < min_area:
            cv2.drawContours(image, [contour], -1, 0, -1)

    return image


def resize_image(img: np.ndarray, new_height: int) -> np.ndarray:
    """
    Resizes an image to a given height while maintaining the aspect ratio.

    Args:
        img (np.ndarray): The input image.
        new_height (int): The desired height for the resized image.

    Returns:
        np.ndarray: The resized image.
    """
    # Calculate the aspect ratio
    aspect_ratio = img.shape[1] / img.shape[0]

    # Calculate the new width
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_img


def fill_small_holes(depth_img: np.ndarray, area_thresh: int) -> np.ndarray:
    """
    Identifies regions in the depth image that have a value of 0 and fills them in
    with 1 if the region is smaller than a given area threshold.

    Args:
        depth_img (np.ndarray): The input depth image
        area_thresh (int): The area threshold for filling in holes

    Returns:
        np.ndarray: The depth image with small holes filled in
    """
    # Create a binary image where holes are 1 and the rest is 0
    binary_img = np.where(depth_img == 0, 1, 0).astype("uint8")

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filled_holes = np.zeros_like(binary_img)

    for cnt in contours:
        # If the area of the contour is smaller than the threshold
        if cv2.contourArea(cnt) < area_thresh:
            # Fill the contour
            cv2.drawContours(filled_holes, [cnt], 0, 1, -1)

    # Create the filled depth image
    filled_depth_img = np.where(filled_holes == 1, 1, depth_img)

    return filled_depth_img


import numpy as np
import torch

def mask_rgb_by_depth_generic(
    rgb,
    depth,
    max_depth: float = 5.0,
    gray_value: float = None,
    mask_when_equal: bool = True
):
    """
    Mask out (replace with gray) all RGB pixels whose corresponding depth is at or beyond `max_depth`.

    This function handles these “generic” cases:
      - Batched or unbatched inputs.
      - NumPy arrays or PyTorch tensors.
      - Channel‐last (H, W, 3) or channel‐first (3, H, W) layouts (and their batched counterparts).

    Parameters
    ----------
    rgb : np.ndarray or torch.Tensor
        The RGB image. Supported shapes:
          - Unbatched:
              * (H, W, 3)       — channel‐last
              * (3, H, W)       — channel‐first
          - Batched:
              * (B, H, W, 3)    — channel‐last
              * (B, 3, H, W)    — channel‐first
        dtype can be uint8, float32, float64, etc. All channels must be in one contiguous array/tensor.
    depth : np.ndarray or torch.Tensor
        The depth image. Supported shapes:
          - Unbatched:
              * (H, W)         — no explicit channel
              * (H, W, 1)      — channel‐last, single‐channel
              * (1, H, W)      — channel‐first, single‐channel
          - Batched:
              * (B, H, W)      — no explicit channel
              * (B, H, W, 1)   — channel‐last, single‐channel
              * (B, 1, H, W)   — channel‐first, single‐channel
        Every pixel must be a distance in meters; values beyond `max_depth` are assumed to have already
        been clipped to exactly `max_depth`.
    max_depth : float, optional (default=5.0)
        The threshold distance (in meters). All pixels whose depth ≥ max_depth (or > max_depth, if
        `mask_when_equal=False`) will be masked in the RGB output.
    gray_value : float or int, optional (default=None)
        The grayscale value to assign to masked RGB pixels. If `None`, the function infers a “middle gray”
        based on the RGB dtype:
          - If dtype is an integer type (e.g. uint8), gray = max_int_value // 2  (e.g. 128 for uint8).
          - If dtype is floating (e.g. float32), gray = 0.5.
        You can also pass a scalar (e.g. 128, 0.5) in which case that value is used for all three channels.
    mask_when_equal : bool, optional (default=True)
        If True, mask pixels with `depth >= max_depth`. If False, mask only those `depth > max_depth`.

    Returns
    -------
    masked_rgb : np.ndarray or torch.Tensor
        An RGB image (same dtype and “layout” as the input `rgb`) where every pixel whose
        corresponding depth is over threshold has been replaced by the chosen gray. The returned shape
        matches exactly the input shape of `rgb`.
    """

    # ------------------------
    # Step 1: Detect backend (NumPy or PyTorch) and validate types
    # ------------------------
    is_numpy = isinstance(rgb, np.ndarray)
    is_torch = isinstance(rgb, torch.Tensor)
    if not (is_numpy or is_torch):
        raise TypeError(f"rgb must be a NumPy array or a torch.Tensor, got {type(rgb)}")

    if not isinstance(depth, type(rgb)):
        raise TypeError("rgb and depth must be the same type (both NumPy arrays or both torch.Tensors)")

    xp = np if is_numpy else torch  # alias for array ops

    # ------------------------
    # Step 2: Record original shapes and detect “layout”
    # ------------------------
    rgb_shape = rgb.shape
    depth_shape = depth.shape

    # ------- RGB layout detection -------
    # Possible rgb shapes:
    #   Unbatched: (H, W, 3) or (3, H, W)
    #   Batched:   (B, H, W, 3) or (B, 3, H, W)
    rgb_ndim = rgb.ndim
    batched_rgb = False
    rgb_channel_first = False

    if rgb_ndim == 4:
        batched_rgb = True
        if rgb_shape[1] == 3:  # (B, 3, H, W)
            rgb_channel_first = True
            B, C_rgb, H, W = rgb_shape
            if C_rgb != 3:
                raise ValueError(f"Unexpected channel size {C_rgb} for RGB. Expected 3.")
        elif rgb_shape[3] == 3:  # (B, H, W, 3)
            rgb_channel_first = False
            B, H, W, C_rgb = rgb_shape
        else:
            raise ValueError(f"Unrecognized 4D shape for RGB: {rgb_shape}")
    elif rgb_ndim == 3:
        batched_rgb = False
        if rgb_shape[0] == 3:  # (3, H, W)
            rgb_channel_first = True
            C_rgb, H, W = rgb_shape
        elif rgb_shape[2] == 3:  # (H, W, 3)
            rgb_channel_first = False
            H, W, C_rgb = rgb_shape
        else:
            raise ValueError(f"Unrecognized 3D shape for RGB: {rgb_shape}")
    else:
        raise ValueError(f"rgb must be 3D or 4D, got shape {rgb_shape}")

    # ------- Depth layout detection -------
    # Possible depth shapes:
    #   Unbatched: (H, W), (H, W, 1), (1, H, W)
    #   Batched:   (B, H, W), (B, H, W, 1), (B, 1, H, W)
    depth_ndim = depth.ndim
    batched_depth = False
    depth_channel_first = False  # single-channel layout if present

    if depth_ndim == 3:
        # Could be unbatched (H, W, 1) or (1, H, W), or batched (B, H, W)
        # Distinguish by checking if any dimension equals 1 (channel) vs. batch dimension:
        if batched_rgb:
            # If rgb is batched, we expect depth to be batched as well.
            # If shape is (B, H, W), treat as batched, no explicit channel.
            if depth_shape[0] == B and depth_shape[1] == H and depth_shape[2] == W:
                batched_depth = True
                depth_channel_first = False
            else:
                # Could be (B, 1, H) or something invalid
                # Actually for depth, (B, 1, H, W) would be 4D; so 3D depth with depth_shape[0]==1 means unbatched.
                batched_depth = False
        if not batched_depth:
            # Unbatched depth. Check for shape (H, W) vs (H, W, 1) vs (1, H, W)
            # If depth_shape == (H, W), that's perfect.
            if depth_shape == (H, W):
                batched_depth = False
                depth_channel_first = False
            elif depth_shape == (H, W, 1):
                batched_depth = False
                depth_channel_first = False
            elif depth_shape == (1, H, W):
                batched_depth = False
                depth_channel_first = True
            else:
                raise ValueError(f"Unrecognized 3D shape for depth: {depth_shape}")
    elif depth_ndim == 4:
        # Must be batched: (B, H, W, 1) or (B, 1, H, W)
        batched_depth = True
        if depth_shape[1] == 1 and depth_shape[2] == H and depth_shape[3] == W:
            depth_channel_first = True  # (B, 1, H, W)
        elif depth_shape[1] == H and depth_shape[2] == W and depth_shape[3] == 1:
            depth_channel_first = False  # (B, H, W, 1)
        elif depth_shape[0] == B and depth_shape[1] == H and depth_shape[2] == W:
            # Actually this is 3D—but we are in the 4D block, so this can't happen
            raise ValueError(f"Impossible 4D shape for depth: {depth_shape}")
        else:
            raise ValueError(f"Unrecognized 4D shape for depth: {depth_shape}")
    elif depth_ndim == 2:
        # Unbatched, simple (H, W)
        batched_depth = False
        depth_channel_first = False
    else:
        raise ValueError(f"depth must be 2D, 3D, or 4D, got shape {depth_shape}")

    # Sanity check: if rgb is batched, depth must be batched with the same B
    if batched_rgb:
        if not batched_depth:
            raise ValueError("rgb is batched but depth is not. Both must be batched or both unbatched.")
        if depth_ndim >= 3:
            if depth_ndim == 3:
                # shape is (B, H, W)
                if depth_shape[0] != B:
                    raise ValueError(f"Batch size mismatch: rgb batch={B}, depth batch={depth_shape[0]}")
            else:  # depth_ndim == 4
                if depth_shape[0] != B:
                    raise ValueError(f"Batch size mismatch: rgb batch={B}, depth batch={depth_shape[0]}")
    else:
        if batched_depth:
            raise ValueError("depth is batched but rgb is not. Both must be batched or both unbatched.")

    # ------------------------
    # Step 3: Canonicalize to “channel‐last, batched if needed” layout
    # ------------------------
    # For RGB: target shape will be (B, H, W, 3) or (H, W, 3) if unbatched.
    # For Depth: target shape will be (B, H, W) or (H, W) if unbatched.
    if is_numpy:
        # --- Handle RGB ---
        if batched_rgb:
            if rgb_channel_first:
                # (B, 3, H, W) → (B, H, W, 3)
                rgb_canon = np.transpose(rgb, (0, 2, 3, 1))
            else:
                # (B, H, W, 3) is already correct
                rgb_canon = rgb.copy()
        else:
            if rgb_channel_first:
                # (3, H, W) → (H, W, 3)
                rgb_canon = np.transpose(rgb, (1, 2, 0))
            else:
                # (H, W, 3) already correct
                rgb_canon = rgb.copy()

        # --- Handle Depth ---
        if batched_depth:
            if depth_ndim == 3:
                # shape (B, H, W) → keep as‐is
                depth_canon = depth.copy()
            else:
                # depth_ndim == 4
                if depth_channel_first:
                    # (B, 1, H, W) → remove channel axis → (B, H, W)
                    depth_canon = depth.squeeze(1)
                else:
                    # (B, H, W, 1) → remove channel axis → (B, H, W)
                    depth_canon = depth.squeeze(3)
        else:
            # Unbatched
            if depth_ndim == 2:
                # (H, W)
                depth_canon = depth.copy()
            elif depth_ndim == 3:
                if depth_channel_first:
                    # (1, H, W) → squeeze → (H, W)
                    depth_canon = depth.squeeze(0)
                else:
                    # (H, W, 1) → squeeze → (H, W)
                    depth_canon = depth.squeeze(2)
            else:
                # Should not reach here
                raise RuntimeError("Unexpected unbatched depth layout")

    else:
        # PyTorch branch
        # --- Handle RGB ---
        if batched_rgb:
            if rgb_channel_first:
                # (B, 3, H, W) → (B, H, W, 3)
                rgb_canon = rgb.permute(0, 2, 3, 1).clone()
            else:
                # (B, H, W, 3) → no change
                rgb_canon = rgb.clone()
        else:
            if rgb_channel_first:
                # (3, H, W) → (H, W, 3)
                rgb_canon = rgb.permute(1, 2, 0).clone()
            else:
                # (H, W, 3) → no change
                rgb_canon = rgb.clone()

        # --- Handle Depth ---
        if batched_depth:
            if depth_ndim == 3:
                # (B, H, W)
                depth_canon = depth.clone()
            else:
                # depth_ndim == 4
                if depth_channel_first:
                    # (B, 1, H, W) → (B, H, W)
                    depth_canon = depth.squeeze(1).clone()
                else:
                    # (B, H, W, 1) → (B, H, W)
                    depth_canon = depth.squeeze(3).clone()
        else:
            # Unbatched
            if depth_ndim == 2:
                # (H, W)
                depth_canon = depth.clone()
            elif depth_ndim == 3:
                if depth_channel_first:
                    # (1, H, W) → (H, W)
                    depth_canon = depth.squeeze(0).clone()
                else:
                    # (H, W, 1) → (H, W)
                    depth_canon = depth.squeeze(2).clone()
            else:
                raise RuntimeError("Unexpected unbatched depth layout")

    # Now:
    #   rgb_canon has shape (B, H, W, 3) if batched, else (H, W, 3)
    #   depth_canon has shape (B, H, W) if batched, else (H, W)

    # ------------------------
    # Step 4: Build boolean mask where depth >= max_depth (or >, if mask_when_equal=False)
    # ------------------------
    # if mask_when_equal:
    #     if is_numpy:
    #         mask = (depth_canon <= max_depth)
    #     else:
    #         mask = depth_canon <= max_depth
    # else:
    #     if is_numpy:
    #         mask = (depth_canon < max_depth)
    #     else:
    #         mask = depth_canon < max_depth

    if mask_when_equal:
        if is_numpy:
            mask = (depth_canon >= max_depth)
        else:
            mask = depth_canon >= max_depth
    else:
        if is_numpy:
            mask = (depth_canon > max_depth)
        else:
            mask = depth_canon > max_depth

    # mask has shape (B, H, W) or (H, W), dtype=bool

    # ------------------------
    # Step 5: Determine gray value (if not provided)
    # ------------------------
    # gray_value can be a scalar; we broadcast it to three channels below.
    if gray_value is None:
        # Infer from dtype
        if is_numpy:
            dtype = rgb.dtype
            if np.issubdtype(dtype, np.integer):
                # e.g. uint8 → max is 255 → gray = 128
                max_int = np.iinfo(dtype).max
                gray_scalar = max_int // 2
                gray_value = dtype.type(gray_scalar)
            else:
                # float types → assume range [0, 1]
                gray_value = dtype.type(0.5)
        else:
            dtype = rgb.dtype
            if dtype.is_floating_point:
                gray_value = rgb.new_tensor(0.5).item()
            else:
                # e.g. torch.uint8 or torch.int types
                info = torch.iinfo(dtype)
                gray_value = int(info.max // 2)
    # At this point, gray_value is a Python scalar (float or int).

    # ------------------------
    # Step 6: Apply mask to RGB (in canonical layout)
    # ------------------------
    # We need a 3‐channel mask of same shape as rgb_canon.
    if batched_rgb:
        B_c, H_c, W_c, _ = rgb_canon.shape
        if is_numpy:
            mask_3ch = np.stack([mask, mask, mask], axis=-1)  # (B, H, W, 3)
            masked_canon = rgb_canon.copy()
            # Replace masked pixels in all three channels with gray_value
            masked_canon[mask_3ch] = gray_value
        else:
            # torch
            mask_3ch = mask.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B, H, W, 3)
            masked_canon = rgb_canon.clone()
            masked_canon[mask_3ch] = gray_value
    else:
        H_c, W_c, _ = rgb_canon.shape
        if is_numpy:
            mask_3ch = np.stack([mask, mask, mask], axis=-1)  # (H, W, 3)
            masked_canon = rgb_canon.copy()
            masked_canon[mask_3ch] = gray_value
        else:
            # torch
            mask_3ch = mask.unsqueeze(-1).expand(-1, -1, 3) if mask.ndim == 3 else mask.unsqueeze(-1).expand(H_c, W_c, 3)
            # Actually for unbatched, mask.ndim==2: so do mask.unsqueeze(-1)
            if mask.ndim == 2:
                mask_3ch = mask.unsqueeze(-1).expand(H_c, W_c, 3)  # (H, W, 3)
            masked_canon = rgb_canon.clone()
            masked_canon[mask_3ch] = gray_value

    # ------------------------
    # Step 7: Convert back to original layout
    # ------------------------
    if is_numpy:
        if batched_rgb:
            if rgb_channel_first:
                # masked_canon is (B, H, W, 3) → go back to (B, 3, H, W)
                masked_rgb = masked_canon.transpose(0, 3, 1, 2)
            else:
                # (B, H, W, 3) is already correct
                masked_rgb = masked_canon
        else:
            if rgb_channel_first:
                # (H, W, 3) → (3, H, W)
                masked_rgb = masked_canon.transpose(2, 0, 1)
            else:
                # (H, W, 3) is already correct
                masked_rgb = masked_canon
    else:
        # PyTorch branch
        if batched_rgb:
            if rgb_channel_first:
                # (B, H, W, 3) → (B, 3, H, W)
                masked_rgb = masked_canon.permute(0, 3, 1, 2).contiguous()
            else:
                # (B, H, W, 3) is already correct
                masked_rgb = masked_canon
        else:
            if rgb_channel_first:
                # (H, W, 3) → (3, H, W)
                masked_rgb = masked_canon.permute(2, 0, 1).contiguous()
            else:
                # (H, W, 3) is already correct
                masked_rgb = masked_canon

    return masked_rgb