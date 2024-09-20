import numpy as np

def computeBilinerWeights(q):

    # Convert the input q to a numpy array if it's not already
    q = np.array(q)
    
    # Calculate the integer coordinates of the bottom-left corner
    x = np.floor(q[0])
    y = np.floor(q[1])
    
    # Calculate the fractional part of qx and qy
    dx = q[0] - x
    dy = q[1] - y
    
    # Compute the weights based on bilinear interpolation formula using numpy
    w0 = (1 - dx) * (1 - dy)  # weight for pixel (x, y)
    w1 = dx * (1 - dy)        # weight for pixel (x + 1, y)
    w2 = (1 - dx) * dy        # weight for pixel (x, y + 1)
    w3 = dx * dy              # weight for pixel (x + 1, y + 1)
    
    # Return the weights as a numpy array
    weights = np.array([w0, w1, w2, w3])
    
    return weights

def computeGaussianWeights(winsize, sigma):
    """
    Computes a matrix of Gaussian weights given the window size and sigma.
    
    Args:
        winsize: A tuple (height, width) representing the size of the window.
        sigma: The standard deviation of the Gaussian distribution.
    
    Returns:
        A 2D numpy array of size (height, width) containing Gaussian weights.
    """
    height, width = winsize
    # Calculate the center of the window
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    
    # Create an empty array to store the weights
    weights = np.zeros((height, width))

    # Calculate the Gaussian weights for each point in the window
    for i in range(height):
        for j in range(width):
            x = j - center_x
            y = i - center_y
            # Apply the 2D Gaussian formula
            weights[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the weights so that they sum to 1
    weights /= np.sum(weights)
    
    return weights

def invertMatrix2x2(A):
    """
    Computes the inverse of a 2x2 matrix.
    
    Args:
        A: A numpy array of shape (2, 2) representing the 2x2 matrix.
    
    Returns:
        invA: A numpy array of shape (2, 2) representing the inverse of A.
    """
    assert A.shape == (2, 2), "Input matrix must be 2x2."
    
    # Calculate the determinant
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    
    # Compute the inverse using the formula
    invA = (1 / det) * np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]])
    
    return invA