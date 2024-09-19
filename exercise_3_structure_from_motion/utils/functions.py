import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Tuple, Union, Dict

# These are typehints, they mostly make the code readable and testable
t_points = np.array
t_camera_parameters = np.array
t_descriptors = np.array
t_homography = np.array
t_view = np.array
t_img = np.array
t_images = Dict[str, t_img]
t_homographies = Dict[Tuple[str, str], t_homography]  # The keys are the keys of src and destination images

np.set_printoptions(edgeitems=30, linewidth=180,
                    formatter=dict(float=lambda x: "%8.05f" % x))

def extract_features(img: t_img, num_features: int = 500) -> Tuple[t_points, t_descriptors]:
    """Extract keypoints and their descriptors.
    The OpenCV implementation of ORB is used as a backend.
    https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF
    fast robust local feature detector, It is based on the FAST keypoint detector and a modified version of the visual descriptor BRIEF (Binary Robust Independent Elementary Features).
    Its aim is to provide a fast and efficient alternative to SIFT.

    Args:
        img: a numpy array of [H x Wx 3] size with byte values.
        num_features: an integer signifying how many points we desire.

    Returns:
        A tuple containing a numpy array of [N x 2] and numpy array of [N x 32]
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(num_features)
    keypoints, desciptors= orb.detectAndCompute(gray_img, None)
    points = cv2.KeyPoint_convert(keypoints)
    return (points, desciptors)

def fast_filter_and_align_descriptors(src: Tuple[t_points, t_descriptors], dst: Tuple[t_points, t_descriptors],
                                      similarity_threshold=0.8) -> Tuple[t_points, t_points]:
    """
    Aligns pairs of keypoints from two images.

    Aligns keypoints from two images based on descriptor similarity using Flann based matcher from cv2
    If K points have been detected in image1 and J points have been detected in image2, the result will be to sets of N
    points representing points with similar descriptors; where N <= J and K <=points.
    Args:
        src: A tuple of two numpy arrays with the first array having dimensions [N x 2] and the second one [N x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        dst: A tuple of two numpy arrays with the first array having dimensions [J x 2] and the second one [J x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        similarity_threshold: The ratio the distance of most similar descriptor to the distance of the second
            most similar.

    Returns:
        A tuple of numpy arrays both sized [N x 2] representing the similar point locations.
    """
    epsilon = .000000001 # for division by zero
    (points1, descriptors1), (points2, descriptors2) = src, dst

    FLANN_INDEX_LSH = 6
    flann_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=3)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    knnmatches = matcher.knnMatch(descriptors1, descriptors2, 2)

    first_to_second_ratios = np.zeros(len(knnmatches))
    indices1 = np.zeros(len(knnmatches), dtype=np.int32)
    indices2 = np.zeros(len(knnmatches), dtype=np.int32)
    for n, (first, second) in enumerate(knnmatches):
        first_to_second_ratios[n] = (first.distance / (second.distance + epsilon))
        indices1[n] = first.queryIdx
        indices2[n] = first.trainIdx

    keep_idx = first_to_second_ratios <= similarity_threshold
    filtered_indices1, filtered_indices2 = indices1[keep_idx], indices2[keep_idx]
    return points1[filtered_indices1, :], points2[filtered_indices2, :]


#student function
def inliers_epipolar_constraint(p1: np.array, p2: np.array, F: np.array, distance_threshold: float) -> np.array:
    """Returns indices of points that satisfy the epipolar constraint"""
    
    # Step 1: Convert points p1 and p2 from 2D to (2+1)D homogeneous coordinates
    p1_h = np.hstack([p1, np.ones((p1.shape[0], 1))])  # Convert p1 to homogeneous coordinates
    p2_h = np.hstack([p2, np.ones((p2.shape[0], 1))])  # Convert p2 to homogeneous coordinates
    
    # Step 2: Compute the epipolar lines for p1 in the second image (using the fundamental matrix)
    epipolar_lines = F @ p1_h.T  # Compute epipolar lines in the second image (shape: [3, N])
    epipolar_lines = epipolar_lines.T  # Transpose to match point dimensions [N, 3]
    
    # Step 3: Normalize the epipolar lines
    epipolar_lines /= np.sqrt(epipolar_lines[:, 0]**2 + epipolar_lines[:, 1]**2).reshape(-1, 1)  # Normalize lines
    
    # Step 4: Compute the perpendicular distances from p2 to the epipolar lines
    distances = np.abs(np.sum(epipolar_lines * p2_h, axis=1))  # Calculate distance
    
    # Step 5: Find inliers based on the distance threshold
    inliers = np.where(distances < distance_threshold)[0]  # Get indices of inliers
    
    return inliers


# student function
import numpy as np

def compute_fundamental_matrix(points1: np.array, points2: np.array) -> np.array:
    """Computes the fundamental matrix given pairs of corresponding points in two images.

    Args:
        points1: A numpy array of size [N x 2] containing the x and y coordinates of the source points.
        points2: A numpy array of size [N x 2] containing the x and y coordinates of the destination points.

    Returns:
        A [3 x 3] numpy array containing the normalized fundamental matrix.
    """
    assert (len(points1) == 8), "Length of points1 should be 8!"
    assert (len(points2) == 8), "Length of points2 should be 8!"

    # Step 1: Construct the 8x9 matrix A
    A = np.zeros((8, 9))
    for i in range(8):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A[i] = [x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1]
    
    # Step 2: Solve Af = 0 using SVD
    # Perform SVD on A
    U, S, Vt = np.linalg.svd(A)
    
    # The solution is the right singular vector corresponding to the smallest singular value
    F = Vt[-1].reshape(3, 3)  # Take the last row of V (corresponds to the smallest singular value)
    
    # Step 3: Enforce Rank(F) = 2
    # Perform SVD on F
    U_F, S_F, Vt_F = np.linalg.svd(F)
    
    # Set the smallest singular value to 0 to enforce rank 2
    S_F[2] = 0
    
    # Recompute F
    F_rank2 = U_F @ np.diag(S_F) @ Vt_F
    
    # Step 4: Normalize the fundamental matrix
    F_normalized = F_rank2 / F_rank2[2, 2]
    
    return F_normalized


def compute_F_ransac(points1: t_points, points2: t_points, distance_threshold: float = 4.0, steps: int = 1000,
                     n_points: int = 8) -> Tuple[np.array, np.array]:
    """Finds the best homography matrix with the RANSAC algorithm.

    Args:
        points1: A numpy array of size [N x 2] containing the x and y coordinates of the source points.
        points2: A numpy array of size [N x 2] containing the x and y coordinates of the destination points.
        distance_threshold: a float representing the norm of the difference between two points so that they will be considered the same (close enough)
        steps: An integer value representing the iteration count for ransac
        n_points: An integer value representing how many points the ransac algorithm should use to compute the homography matrix

    Returns:
        A tuple with the best homography matrix found and its corresponding inlier indices.
    """
    best_count = 0
    best_homography = np.eye(3)
    best_inlier_indices = np.array([])

    for n in tqdm(range(steps)):
        if n == steps - 1:
            print(f"Step: {n:4}  {best_count} RANSAC points match!")

        randomidx = np.random.permutation(points1.shape[0])[:n_points]
        rnd_points1, rnd_points2 = points1[randomidx, :], points2[randomidx, :]

        homography = compute_fundamental_matrix(rnd_points1, rnd_points2)
        inliers_indices = inliers_epipolar_constraint(points1, points2, homography, distance_threshold)

        if inliers_indices.shape[0] > best_count:
            best_count = inliers_indices.shape[0]
            best_homography = homography
            best_inlier_indices = inliers_indices
    return best_homography, np.array(best_inlier_indices)


# student function


def triangulate(P1: np.array, P2: np.array, p1: np.array, p2: np.array) -> np.array:
    """Projects a point from its location in the two images to "real-world" 3D coordinates.

    Args:
        P1: A [3 x 4] numpy array representing the camera matrix for View 1 (camera 1 is at 0, 0, 0)
        P2: A [3 x 4] numpy array representing the camera matrix for View 2
        p1: A numpy array of shape [2] representing a single point from image 1
        p2: A numpy array of shape [2] representing a single point from image 2
    Returns:
        resulting_point: A numpy array of shape [3] representing the point in 3D space
    """
    epsilon = 1e-8  # to avoid division by zero

    # Step 1: Construct the matrix A
    A = np.zeros((4, 4))
    A[0] = p1[0] * P1[2, :] - P1[0, :]
    A[1] = p1[1] * P1[2, :] - P1[1, :]
    A[2] = p2[0] * P2[2, :] - P2[0, :]
    A[3] = p2[1] * P2[2, :] - P2[1, :]
    
    # Step 2: Solve for X using SVD
    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]  # Right singular vector corresponding to the smallest singular value
    
    # Step 3: Convert back from homogeneous to 3D (Euclidean coordinates)
    X_homogeneous /= X_homogeneous[-1] + epsilon  # Avoid division by zero by adding a small epsilon
    X_3D = X_homogeneous[:3]  # Extract the [x, y, z] coordinates

    return X_3D



def triangulate_all_points(View1: t_view, View2: t_view, K: t_view, points1: t_points, points2: t_points) \
        -> t_points:
    """Creates a 3D pointcloud out of corresponding points in two images.

    The pointcloud is also filtered from outliers (points estimated to occur behind the cameras).

    Args:
        View1: A numpy array of shape [3 x 4] representing View matrix 1 for Camera
        View2: A numpy array of shape [3 x 4] representing View matrix 2 for Camera
        K: A numpy array of shape [3 x 3] representing the intrinsic camera parameters
        points1: A numpy array of size [N x 2] containing x and y coordinates of the source points.
        points2: A numpy array of size [N x 2] containing x and y coordinates of the destination points.

    Returns:
        wps: A numpy array of shape [N x 3] representing the point cloud in 3D space
    """
    wps = []
    P1 = np.dot(K, View1)
    P2 = np.dot(K, View2)
    for i in range(len(points1)):
        wp = triangulate(P1, P2, points1[i], points2[i])
        # Check if this points is in front of both cameras
        ptest = [wp[0], wp[1], wp[2], 1]
        p1 = np.matmul(P1, ptest)
        p2 = np.matmul(P2, ptest)
        if (p1[2] > 0) and (p2[2] > 0):
            wps.append(wp)
    wps = np.array(wps)
    return wps


# student function
import numpy as np

def compute_essential_matrix(fundamental_matrix: np.array, camera_parameters: np.array) -> np.array:
    """Computes the essential matrix given the fundamental matrix and the intrinsic camera parameters

    Args:
        fundamental_matrix: A [3 x 3] numpy array representing the fundamental matrix
        camera_parameters: A [3 x 3] numpy array representing the intrinsic camera parameters

    Returns:
        essential_matrix: A [3 x 3] numpy array representing the essential matrix
    """
    # Step 1: Compute the essential matrix using E = K^T * F * K
    essential_matrix = camera_parameters.T @ fundamental_matrix @ camera_parameters
    
    return essential_matrix


# student function


def decompose(E: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
    """Decomposes an essential matrix into the two rotations and 2 translations that constitute it.

    Args:
        E: A [3 x 3] numpy array representing the essential matrix.

    Returns:
        A tuple of R1, R2, t1, t2. R1, R2 are two rotation matrices of shape [3 x 3] and t1, t2 are two translation
            matrices of shape [3]
    """
    # Step 1 - Compute the SVD of E
    U, _, Vt = np.linalg.svd(E)

    # Ensure that U and Vt are proper rotation matrices (determinants should be +1)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Step 2 - Compute W and its transpose
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    # Step 3 - Compute the two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Step 4 - Compute the two possible translations (third column of U)
    t1 = U[:, 2]
    t2 = -U[:, 2]

    return R1, R2, t1, t2



def relativeTransformation(E: t_homography, points1: t_points, points2: t_points, K: t_camera_parameters) -> t_view:
    """Constructs the View Camera Matrix for the second camera position.

    Args:
        E: A [3 x 3] numpy array representing the essential matrix
        points1: A numpy array of size [N x 2] containing x and y coordinates of the source points.
        points2: A numpy array of size [N x 2] containing x and y coordinates of the destination points.
        K: A numpy array of shape [3 x 3] representing the intrinsic camera parameters

    Returns:
        V: A numpy array of shape [3 x 4] representing View matrix 2 for Camera
    """

    R1, R2, t1, t2 = decompose(E)
    ## A negative determinant means that R contains a reflection.This is not rigid transformation!
    if np.linalg.det(R1) < 0:
        E = -E
        R1, R2, t1, t2 = decompose(E)

    bestCount = 0

    for dR in range(2):
        if dR == 0:
            cR = R1
        else:
            cR = R2

        for dt in range(2):
            if dt == 0:
                ct = t1
            else:
                ct = t2

            View1 = np.eye(3, 4)
            View2 = np.zeros((3, 4))
            for i in range(3):
                for j in range(3):
                    View2[i, j] = cR[i, j]
            for i in range(3):
                View2[i, 3] = ct[i]

            count = len(triangulate_all_points(View1, View2, K, points1, points2))
            if (count > bestCount):
                V = View2
                bestCount = count

    return V
