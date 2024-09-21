import cv2
import sys
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.spatial as spatial
from typing import Tuple, Dict, List
from scipy.spatial.distance import cdist

# These are type hints, they mostly make the code readable and testable
t_img = np.array
t_disparity = np.array
t_points = np.array
t_descriptors = np.array


def extract_features(img: t_img, num_features: int = 500) -> Tuple[t_points, t_descriptors]:
    """Extracts key-points and their descriptors.
    The OpenCV implementation of ORB is used as a backend.
    It is based on the FAST key-point detector and a modified version of the visual descriptor BRIEF (Binary Robust Independent Elementary Features).
    Its aim is to provide a fast and efficient alternative to SIFT.

    Args:
        img: a numpy array of [H x Wx 3] size with byte values.
        num_features: an integer signifying how many points we desire.

    Returns:
        A tuple containing a numpy array of [N x 2] and numpy array of [N x 32]
    """
    #TODO : Hint - you will need cv2.ORB_create
      # Step 1: Initialize the ORB detector with the desired number of features
    orb = cv2.ORB_create(nfeatures=num_features)

    # Step 2: Convert the image to grayscale if it is not already
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Step 3: Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)

    # Step 4: Convert keypoints to numpy array of [N x 2] format
    keypoint_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

    # Return the keypoints and descriptors
    return keypoint_array, descriptors




def filter_and_align_descriptors(f1: Tuple[t_points, t_descriptors], f2: Tuple[t_points, t_descriptors],
                                 similarity_threshold=.7, similarity_metric='hamming') -> Tuple[t_points, t_points]:
    """Aligns pairs of keypoints from two images.
    Aligns keypoints from two images based on descriptor similarity.
    If K points have been detected in image1 and J points have been detected      in image2, the result will be to sets of N
    points representing points with similar descriptors; where N <= J and K <=points.

    Args:
        f1: A tuple of two numpy arrays with the first array having dimensions [N x 2] and the second one [N x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        f2: A tuple of two numpy arrays with the first array having dimensions [J x 2] and the second one [J x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        similarity_threshold: The ratio the distance of most similar descriptor in image2 to the distance of the second
            most similar ratio.
        similarity_metric: A string with the name of the metric by witch distances are calculated. It must be compatible
            with the ones that are defined for scipy.spatial.distance.cdist.

    Returns:
        A tuple of numpy arrays both sized [N x 2] representing the similar point locations.

    """
    assert f1[0].shape[1] == f2[0].shape[1] == 2  # descriptor size
    assert f1[1].shape[1] == f2[1].shape[1] == 32  # points size

    # Step 1: Compute the distance matrix between descriptors in f1 and f2
    distance_matrix = cdist(f1[1], f2[1], metric=similarity_metric)

    # Step 2: Find the indices of the best and second-best matches for each descriptor in f1
    best_match_indices = np.argmin(distance_matrix, axis=1)
    sorted_distances = np.sort(distance_matrix, axis=1)
    best_distances = sorted_distances[:, 0]
    second_best_distances = sorted_distances[:, 1]

    # Step 3: Apply the ratio test to find significant matches
    ratio_mask = (best_distances / (second_best_distances + 1e-10)) < similarity_threshold

    # Step 4: Filter out the non-significant matches
    src_points = f1[0][ratio_mask]
    dst_points = f2[0][best_match_indices[ratio_mask]]

    return src_points, dst_points


def get_max_translation(src: t_img, dst: t_img, well_aligned_thr=.1) -> int:
    """Finds the maximum translation/shift between two images.

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        well_aligned_thr: a float representing the maximum y-wise distance between valid matching points.

    Returns:
        An integer value representing the maximum translation of the camera from src to dst image.
    """

    # Step 1: Extract features/descriptors from both images
    f1 = extract_features(src)  # Features from src image
    f2 = extract_features(dst)  # Features from dst image

    # Step 2: Filter and align descriptors
    src_points, dst_points = filter_and_align_descriptors(f1, f2)

    # Step 3: Filter out correspondences that are not horizontally aligned using well-aligned threshold
    # Keep matches where the y-coordinate difference between src and dst points is below the threshold
    y_thr = well_aligned_thr * src.shape[0]  # Threshold scaled by the image height
    well_aligned_mask = np.abs(src_points[:, 1] - dst_points[:, 1]) <= y_thr

    # Apply the well-aligned mask to keep only horizontally aligned points
    well_aligned_src_points = src_points[well_aligned_mask]
    well_aligned_dst_points = dst_points[well_aligned_mask]

    # Step 4: Find the translation across the image using the descriptors and return the maximum value
    if well_aligned_src_points.shape[0] == 0:
        raise ValueError("No well-aligned points found.")

    # Compute the horizontal translations (x-axis differences)
    translations = well_aligned_dst_points[:, 0] - well_aligned_src_points[:, 0]

    # Return the maximum absolute translation (rounded to the nearest integer)
    return int(max(translations, key=abs))

def render_disparity_hypothesis(src: t_img, dst: t_img, offset: int, pad_size: int) -> t_disparity:
    """Calculates the agreement between the shifted src image and the dst image.

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation

    Returns:
        a numpy array of shape [H x W] containing the euclidean distance between RGB values of the shifted src and dst
        images.
    """

    # Step 1: Pad necessary values to src and dst
    # Pad the images to allow shifting without losing information
    src_padded = np.pad(src, ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    dst_padded = np.pad(dst, ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)

    # Step 2: Shift the src image horizontally by the offset value
    shifted_src = np.roll(src_padded, shift=offset, axis=1)

    # Step 3: Crop the images back to the original dimensions (remove padding)
    shifted_src_cropped = shifted_src[:, pad_size: pad_size + src.shape[1], :]
    dst_cropped = dst_padded[:, pad_size: pad_size + dst.shape[1], :]

    # Step 4: Compute the Euclidean distance between corresponding pixels in the shifted src and dst images
    disparity = np.linalg.norm(shifted_src_cropped - dst_cropped, axis=2)

    # Return the disparity map (H x W)
    return disparity


def disparity_map(src: t_img, dst: t_img, offset: int, pad_size: int, sigma_x: int, sigma_z: int,
                  median_filter_size: int) -> t_disparity:
    """calculates the best/minimum disparity map for a given pair of images

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation
        sigma_x: an integer value for standard deviation in x-direction for gaussian filter
        sigma_z: an integer value for standard deviation in z-direction for gaussian filter
        median_filter_size: an integer value representing the window size for applying median filter

    Returns:
        a numpy array of shape [H x W] containing the minimum/best disparity values for a pair of images
    """

    # Step 1: Construct a stack of all reasonable disparity hypotheses.
    # We'll try a range of offsets (disparities), creating a disparity hypothesis for each shift.
    max_disparity = offset  # Assume this as the maximum disparity for now
    H, W, _ = src.shape
    disparity_stack = np.zeros((H, W, max_disparity), dtype=np.float32)
    
    for d in range(max_disparity):
        # For each disparity, we shift the src image and calculate the disparity hypothesis
        disparity_stack[:, :, d] = render_disparity_hypothesis(src, dst, d, pad_size)

    # Step 2: Enforce the coherence between x-axis and disparity-axis using a 3D gaussian filter
    # Apply Gaussian filter on the disparity stack to smooth it across both x (spatial) and z (disparity) dimensions
    disparity_stack = scipy.ndimage.gaussian_filter(disparity_stack, sigma=(sigma_x, sigma_x, sigma_z))

    # Step 3: Choose the best disparity hypothesis for every pixel
    # We choose the disparity that minimizes the disparity value (best match)
    best_disparity = np.argmin(disparity_stack, axis=2)

    # Step 4: Apply the median filter to enhance local consensus
    best_disparity = scipy.ndimage.median_filter(best_disparity, size=median_filter_size)

    return best_disparity


def bilinear_grid_sample(img: t_img, x_array: t_img, y_array: t_img) -> t_img:
    """Sample an image according to a sampling vector field using bilinear interpolation.

    Args:
        img: one image, numpy array of shape [H x W x 3]
        x_array: a numpy array of [H' x W'] representing the x coordinates for interpolation in the x-direction
        y_array: a numpy array of [H' x W'] representing the y coordinates for interpolation in the y-direction

    Returns:
        An image of size [H' x W' x 3] containing the sampled points with bilinear interpolation.
    """
    # Get image dimensions
    H, W, C = img.shape
    H_new, W_new = x_array.shape

    # Initialize output image
    output = np.zeros((H_new, W_new, C), dtype=img.dtype)

    # Step 1: Compute integer pixel positions (left, right, top, bottom) for bilinear interpolation
    x_left = np.floor(x_array).astype(np.int32)
    x_right = x_left + 1
    y_top = np.floor(y_array).astype(np.int32)
    y_bottom = y_top + 1

    # Compute the fractional part for bilinear interpolation
    a = x_array - x_left  # fractional distance from left to right
    b = y_array - y_top   # fractional distance from top to bottom

    # Step 2: Handle boundary conditions by clamping the indices
    x_left = np.clip(x_left, 0, W - 1)
    x_right = np.clip(x_right, 0, W - 1)
    y_top = np.clip(y_top, 0, H - 1)
    y_bottom = np.clip(y_bottom, 0, H - 1)

    # Step 3: Perform bilinear interpolation for each color channel
    for c in range(C):
        # Top-left pixel
        top_left = img[y_top, x_left, c]
        # Top-right pixel
        top_right = img[y_top, x_right, c]
        # Bottom-left pixel
        bottom_left = img[y_bottom, x_left, c]
        # Bottom-right pixel
        bottom_right = img[y_bottom, x_right, c]

        # Step 4: Compute weighted sum of the four corners
        output[:, :, c] = (1 - a) * (1 - b) * top_left + \
                          a * (1 - b) * top_right + \
                          (1 - a) * b * bottom_left + \
                          a * b * bottom_right

    return output

#
