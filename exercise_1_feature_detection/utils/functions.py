import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute Idx and Idy with cv2.Sobel
    Idx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    Idy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)

    # Step 2: Ixx Iyy Ixy from Idx and Idy
    A = Idx * Idx  # Ixx
    B = Idx * Idy  # Ixy
    C = Idy * Idy  # Iyy

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur
    # Use sdev = 1 and kernelSize = (3, 3) in cv2.GaussianBlur
    A = cv2.GaussianBlur(A, (3, 3), sigmaX=1)
    B = cv2.GaussianBlur(B, (3, 3), sigmaX=1)
    C = cv2.GaussianBlur(C, (3, 3), sigmaX=1)

    # Step 4: Compute the harris response with the determinant and the trace of T
    detM = (A * C) - (B * B)
    traceM = A + C
    R = detM - k * (traceM ** 2)

    return R, A, B, C, Idx, Idy


import numpy as np
from typing import Tuple

def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the Harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the Harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1: Pad the response image to facilitate vectorization
    padded_R = np.pad(R, pad_width=1, mode='constant', constant_values=-np.inf)

    # Step 2: Create one image for every offset in the 3x3 neighborhood
    offsets = [
        padded_R[0:-2, 0:-2],  # top-left
        padded_R[0:-2, 1:-1],  # top-center
        padded_R[0:-2, 2:],    # top-right
        padded_R[1:-1, 0:-2],  # middle-left
        padded_R[1:-1, 2:],    # middle-right
        padded_R[2:, 0:-2],    # bottom-left
        padded_R[2:, 1:-1],    # bottom-center
        padded_R[2:, 2:]       # bottom-right
    ]

    # Step 3: Compute the greatest neighbor of every pixel
    max_neighbors = np.maximum.reduce(offsets)

    # Step 4: Compute a boolean image with only key-points set to True
    key_points_mask = (R > threshold) & (R > max_neighbors)

    # Step 5: Use np.nonzero to compute the locations of the key-points from the boolean image
    keypoint_y, keypoint_x = np.nonzero(key_points_mask)

    return keypoint_x, keypoint_y



import numpy as np

def detect_edges(R: np.array, edge_threshold: float = -0.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1: Pad the response image to facilitate vectorization
    padded_R = np.pad(R, pad_width=1, mode='constant', constant_values=np.inf)

    # Step 2: Calculate significant response pixels
    significant = R < edge_threshold

    # Step 3: Create two images with the smaller x-axis and y-axis neighbors respectively
    # Get the left and right neighbors for the x-axis check
    left_neighbor = padded_R[1:-1, :-2]
    right_neighbor = padded_R[1:-1, 2:]

    # Get the top and bottom neighbors for the y-axis check
    top_neighbor = padded_R[:-2, 1:-1]
    bottom_neighbor = padded_R[2:, 1:-1]

    # Step 4: Calculate pixels that are lower than either their x-axis or y-axis neighbors
    x_axis_minimal = (R < left_neighbor) & (R < right_neighbor)
    y_axis_minimal = (R < top_neighbor) & (R < bottom_neighbor)

    # Step 5: Calculate valid edge pixels by combining significant and axis_minimal pixels
    edge_pixels = significant & (x_axis_minimal | y_axis_minimal)

    return edge_pixels

