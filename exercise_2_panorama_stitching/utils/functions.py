import numpy
import cv2
from typing import Tuple, Dict, List
import numpy as np
import scipy.spatial as spatial
from itertools import product
from scipy.linalg import svd
import os
import random
from scipy.spatial.distance import cdist


# These are type hints, they mostly make the code readable and testable
t_points = np.array
t_descriptors = np.array
t_homography = np.array
t_img = np.array
t_images = Dict[str, t_img]
t_homographies = Dict[Tuple[str, str], t_homography]  # The keys are the keys of src and destination images
t_image_list = List[np.array]
t_str_list = List[str]

np.set_printoptions(edgeitems=30, linewidth=180,
                    formatter=dict(float=lambda x: "%8.05f" % x))


def show_images(images: t_image_list, names: t_str_list) -> None:
    """Shows one or more images at once.

    Displaying a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        names: A list of strings that will appear as the window titles for each image

    Returns:
        None
    """
    for image_index in range(0, len(images)):
        cv2.imshow(names[image_index], images[image_index])
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    

def save_images(images: t_image_list, filenames: t_str_list, **kwargs) -> None:
    """Saves one or more images at once.

    Saving a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        filenames: A list of strings where each respective file will be created

    Returns:
        None
    """
    for image_index in range(0, len(images)):
        file_name = filenames[image_index]
        _create_directory(os.path.dirname(file_name))

        cv2.imwrite(file_name, images[image_index])


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


def compute_homography(f1: np.array, f2: np.array) -> np.array:
    """Computes the homography matrix given matching points.

    Args:
        f1: A numpy array of size [N x 2] containing x and y coordinates of the source points.
        f2: A numpy array of size [N x 2] containing x and y coordinates of the destination points.

    Returns:
        A [3 x 3] numpy array containing the normalized homography matrix.
    """
    assert f1.shape[0] == f2.shape[0] >= 4  # Ensure there are at least 4 points

    # Number of points
    N = f1.shape[0]

    # Construct the A matrix (2 rows per point pair)
    A = np.zeros((2 * N, 9))
    
    for i in range(N):
        x, y = f1[i]
        x_prime, y_prime = f2[i]
        
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x_prime * x, x_prime * y, x_prime]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime]

    # Step 2: Compute the SVD of A
    U, S, Vt = svd(A)
    
    # Step 3: The solution to Ah = 0 is the rightmost column of V (or row of Vt)
    h = Vt[-1, :]  # This is the vector containing h0...h8
    
    # Step 4: Reshape h into the homography matrix H
    H = h.reshape(3, 3)
    
    # Step 5: Normalize the matrix H so that h_8 = 1 (i.e., H[2, 2] = 1)
    H = H / H[2, 2]
    
    return H


def _get_inlier_count(src_points: np.array, dst_points: np.array, homography: np.array,
                      distance_threshold: float) -> int:
    """Computes the number of inliers for a homography given aligned points.

    Args:
        src_points: a numpy array of [N x 2] containing source points.
        dst_points: a numpy array of [N x 2] containing destination points.
        homography: a [3 x 3] numpy array representing the homography matrix.
        distance_threshold: a float representing the norm of the difference between two points so that they are
            considered the same (near enough).

    Returns:
        An integer counting how many transformed source points matched destination points.
    """
    assert src_points.shape[1] == dst_points.shape[1] == 2
    assert src_points.shape[0] == dst_points.shape[0]

    # Step 1: Convert source points to homogeneous coordinates (from [x, y] to [x, y, 1])
    src_homogeneous = np.hstack([src_points, np.ones((src_points.shape[0], 1))])

    # Step 2: Project the source points to the destination space using the homography matrix
    projected_points_homogeneous = np.dot(src_homogeneous, homography.T)

    # Step 3: Re-normalize the projected points (convert [x', y', w'] to [x'/w', y'/w'])
    projected_points = projected_points_homogeneous[:, :2] / projected_points_homogeneous[:, 2][:, np.newaxis]

    # Step 4: Compute distances between the projected points and the actual destination points
    distances = np.linalg.norm(projected_points - dst_points, axis=1)

    # Count inliers where distance is less than the threshold
    inlier_count = np.sum(distances < distance_threshold)

    return inlier_count


def ransac(src_features: Tuple[t_points, t_descriptors], dst_features: Tuple[t_points, t_descriptors], steps: int,
           distance_threshold: float, n_points=4, similarity_threshold=.7) -> np.array:
    """Computes the best homography given noisy point descriptors using RANSAC.

    Args:
        src_features: A tuple with points and their descriptors detected in the source image.
        dst_features: A tuple with points and their descriptors detected in the destination image.
        steps: An integer defining how many iterations to run.
        distance_threshold: A float defining how far two points should be to be considered the same.
        n_points: The number of point pairs used to compute the homography, it must be greater than 3.
        similarity_threshold: The ratio of the most similar descriptor to the second most similar in order to consider
            that descriptors from the two images match.

    Returns:
        A numpy array containing the best homography.
    """

    # Step 1: Filter and align descriptors
    aligned_src_points, aligned_dst_points = filter_and_align_descriptors(
        src_features, dst_features, similarity_threshold=similarity_threshold
    )

    # Step 2: Initialize variables for the RANSAC loop
    best_count = 0
    best_homography = np.eye(3)  # Identity matrix as the initial homography

    # Step 3: Optimization loop
    for n in range(steps):

        # Step 3a: Randomly select a subset of matching points (at least n_points)
        idx = random.sample(range(aligned_src_points.shape[0]), n_points)
        subset_src_points = aligned_src_points[idx]
        subset_dst_points = aligned_dst_points[idx]

        # Step 3b: Compute the homography for the random subset of points
        try:
            homography = compute_homography(subset_src_points, subset_dst_points)
        except np.linalg.LinAlgError:
            # In case the homography computation fails due to singularities, skip this iteration
            continue

        # Step 3c: Count the number of inliers using the current homography
        inlier_count = _get_inlier_count(aligned_src_points, aligned_dst_points, homography, distance_threshold)

        # Update the best homography if this one has more inliers
        if inlier_count > best_count:
            best_count = inlier_count
            best_homography = homography

    print(f"After {steps:4} steps: {best_count} RANSAC points match!")

    # Step 4: Return the best homography found
    return best_homography



def propagate_homographies(homographies: t_homographies, reference_name: str) -> t_homographies:
    """Computes homographies from every image to the reference image given a homographies between all pairs of
    consecutive images.

    This method could be loosely described as applying Dijkstra's algorithm applied to exploit the commutative
    relationship of matrix multiplication and compute homography matrices between all images and any image.

    Args:
        homographies: A dictionary where the keys are tuples with the names of each image pair and the values are
            [3 x 3] arrays containing the homographies between those images.
        reference_name: The of the image which will be the destination for all homographies.

    Returns:
        A dictionary of the same form as the input mappning all images to the reference.
    """
    initial = {k: v for k, v in homographies.items()}  # deep copy
    for k, h in list(initial.items()):
        initial[(k[1], k[0])] = np.linalg.inv(h)
    initial[(reference_name, reference_name)] = np.eye(3)  # Added the identity homography for the reference
    desired = set([(k[0], reference_name) for k in homographies.keys()])
    solved = {k: v for k, v in initial.items() if k[1] == reference_name}
    while not (set(solved.keys()) >= desired):

        new_steps = set([(i, s) for i, s in product(initial.keys(), solved.keys()) if
                     s[1] != i[0] and s[0] == i[1] and s[0] != s[1] and (i[0], s[1]) not in solved.keys()])
        # s[1] != i[0] no pair who's product leads to identity
        # s[0] == i[1] only connected pairs
        # s[0]!=s[1] no identity in the solution
        # set removes duplicates

        assert len(new_steps) > 0  # not all desired can be linked to reference
        for initial_k, solved_k in new_steps:
            new_key = initial_k[0], solved_k[1]
            solved[solved_k]
            initial[initial_k]
            solved[new_key] = np.matmul(solved[solved_k], initial[initial_k])
    return solved


def compute_panorama_borders(images: t_images, homographies: t_homographies) -> Tuple[float, float, float, float]:
    """Computes the bounding box of the panorama defined the images and the homographies mapping them to the reference.

    This bounding box can have non integer and even negative coordinates.

    Args:
        images: A dictionary mapping image names to numpy arrays containing images.
        homographies:  A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            mapping from the first image to the second.

    Returns:
        A tuple containing the bounding box [left, top, right, bottom] of the whole panorama if stiched.

    """
    homographies = {k[0]: v for k, v in homographies.items()}  # assining homographies to their source image
    assert homographies.keys() == images.keys()  # map homographies to source image only
    all_corners = []
    for name in sorted(images.keys()):
        img, homography = images[name], homographies[name]
        width, height = img.shape[0], img.shape[1]
        corners = ((0, 0), (0, width), (height, width), (height, 0))
        corners = np.array(corners, dtype='float32')
        all_corners.append(cv2.perspectiveTransform(corners[None, :, :], homography)[0, :, :])
    all_corners = np.concatenate(all_corners, axis=0)
    left, right = np.floor(all_corners[:, 0].min()), np.ceil(all_corners[:, 0].max())
    top, bottom = np.floor(all_corners[:, 1].min()), np.ceil(all_corners[:, 1].max())
    return left, top, right, bottom



def translate_homographies(homographies: t_homographies, dx: float, dy: float) -> t_homographies:
    """Applies a uniform translation to a dictionary with homographies.

    Args:
        homographies: A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            mapping from the first image to the second.
        dx: a float representing the horizontal displacement of the translation.
        dy: a float representing the vertical displacement of the translation.

    Returns:
        A copy of the homographies dict which maps the same keys to the translated matrices.
    """
    # Step 1: Create the translation matrix
    translation_matrix = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])

    # Step 2: Apply the translation to each homography in the dictionary
    translated_homographies = {}
    for key, homography in homographies.items():
        # Apply translation by multiplying the translation matrix with the homography
        translated_homographies[key] = np.dot(translation_matrix, homography)

    return translated_homographies


def stitch_panorama(images: t_images, homographies: t_homographies, output_size: Tuple[int, int],
                   rendering_order: List[str] = []) -> t_images:
    """Stiches images after it reprojects them with a homography.

    Args:
        images: A dictionary mapping image names to numpy arrays containing images.
        homographies: A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            mapping from the first image to the reference image.
        output_size: A tuple with integers representing the witdh and height of the resulting panorama.
        rendering_order: A list containing the names of the images representing the order in witch the images will be
            overlaid. The list must contain either all images names in some permutation or be empty in which case, the
            images will be rendered in the alphanumeric order of their names.
    Returns:
        A numpy array with the panorama image.
    """
    homographies = {k[0]: v for k, v in homographies.items()}  # assining homographies to their source image
    assert homographies.keys() == images.keys()
    if rendering_order == []:
        rendering_order = sorted(images.keys())
    panorama = np.zeros([output_size[1], output_size[0], 3], dtype=np.uint8)
    for name in rendering_order:
        rgba_img = cv2.cvtColor(images[name], cv2.COLOR_RGB2RGBA)
        rgba_img[:, :, 3] = 255
        tmp = cv2.warpPerspective(rgba_img, homographies[name], output_size, cv2.INTER_LINEAR_EXACT)
        new_pixels = ((tmp[:, :, 3] == 255)[:, :, None] & (panorama == np.zeros([1, 1, 3])))
        old_pixels = 1 - new_pixels
        panorama[:, :, :] = panorama * old_pixels + tmp[:, :, :3] * new_pixels
    return panorama


def create_stitched_image(images: t_images, homographies: t_homographies, reference_name: str,
                          rendering_order: List[str] = []):
    """Will create a panorama by stitching the input images after reprojecting them.

    Args:
        images: A dictionary mapping image names to numpy arrays containing images.
        homographies: A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            that can reproject the first image to be aligned with the reference image.
        reference_name: A string with the name of the image to which all other images will be aligned.
        rendering_order: A list containing the names of the images representing the order in witch the images will be
            overlaid. The list must contain either all images names in some permutation or be empty in which case, the
            images will be rendered in the alphanumeric order of their names.
    Returns:
        A numpy array with the panorama image.
    """
    #  from homographies between consecutive images we compute all homographies from any image to the reference.
    homographies = propagate_homographies(homographies, reference_name=reference_name)
    #  lets calculate the panorama size
    left, top, right, bottom = compute_panorama_borders(images, homographies)
    width = int(1 + np.ceil(right) - np.floor(left))
    height = int(1 + np.ceil(bottom) - np.floor(top))
    #  lets make the homographies translate all images inside the panorama.
    homographies = translate_homographies(homographies, -left, -top)
    return stitch_panorama(images, homographies, (width, height), rendering_order=rendering_order)


def _create_directory(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        print(f"Error: {dir_path} - {e.strerror}")
