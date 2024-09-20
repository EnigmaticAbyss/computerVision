import numpy as np
import cv2

from utils.util import *


class OpticalFlowLK:

    def __init__(self, winsize, epsilon, iterations):

        self.winsize = winsize
        self.epsilon = epsilon
        self.iterations = iterations

    def compute(self, prevImg, nextImg, prevPts):
        assert prevImg.size != 0 and nextImg.size != 0, "check prevImg and nextImg"
        assert prevImg.shape[0] == nextImg.shape[0], "size mismatch, rows."
        assert prevImg.shape[1] == nextImg.shape[1], "size mismatch, cols."

        N = prevPts.shape[0]
        status = np.ones(N, dtype=int)
        nextPts = np.copy(prevPts)

        # Compute the spatial derivatives of prev using the Scharr function
        prevDerivx = cv2.Scharr(prevImg, cv2.CV_64F, 1, 0, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        prevDerivy = cv2.Scharr(prevImg, cv2.CV_64F, 0, 1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

        halfWin = np.array([(self.winsize[0] - 1) * 0.5, (self.winsize[1] - 1) * 0.5])
        weights = computeGaussianWeights(self.winsize, 0.3)  # Precomputed Gaussian weights

        for ptidx in range(N):
            u0 = prevPts[ptidx]
            u0 -= halfWin

            u = u0
            iu0 = [int(np.floor(u0[0])), int(np.floor(u0[1]))]

            if iu0[0] < 0 or iu0[0] + self.winsize[0] >= prevImg.shape[1] - 1 or \
                    iu0[1] < 0 or iu0[1] + self.winsize[1] >= prevImg.shape[0] - 1:
                status[ptidx] = 0
                continue

            bw = computeBilinerWeights(u0)  # Compute bilinear weights

            bprev = np.zeros((self.winsize[0] * self.winsize[1], 1))
            A = np.zeros((self.winsize[0] * self.winsize[1], 2))
            AtWA = np.zeros((2, 2))

            idx = 0
            for y in range(self.winsize[1]):
                for x in range(self.winsize[0]):
                    gx = int(iu0[0] + x)
                    gy = int(iu0[1] + y)

                    # Get image gradients from Scharr derivatives
                    Ix = prevDerivx[gy, gx]  # Gradient in x
                    Iy = prevDerivy[gy, gx]  # Gradient in y

                    # Get brightness in the patch
                    bprev[idx, 0] = prevImg[gy, gx]

                    # Fill the gradient matrix A
                    A[idx, 0] = Ix  # Gradient in x direction
                    A[idx, 1] = Iy  # Gradient in y direction

                    # Compute AtWA (Weighted by bilinear interpolation)
                    weight = bw[idx % 4]  # Assuming bw has 4 bilinear weights
                    AtWA[0, 0] += weight * A[idx, 0] * A[idx, 0]
                    AtWA[0, 1] += weight * A[idx, 0] * A[idx, 1]
                    AtWA[1, 0] += weight * A[idx, 1] * A[idx, 0]
                    AtWA[1, 1] += weight * A[idx, 1] * A[idx, 1]

                    idx += 1

            # Compute the inverse of AtWA
            invAtWA = invertMatrix2x2(AtWA)

            # Estimate the target point with the previous point
            u = u0

            # Iterative solver
            for j in range(self.iterations):
                iu = [int(np.floor(u[0])), int(np.floor(u[1]))]

                if iu[0] < 0 or iu[0] + self.winsize[0] >= prevImg.shape[1] - 1 \
                        or iu[1] < 0 or iu[1] + self.winsize[1] >= prevImg.shape[0] - 1:
                    status[ptidx] = 0
                    break

                bw = computeBilinerWeights(u)
                AtWbnbp = np.array([0, 0])

                idx = 0
                for y in range(self.winsize[1]):
                    for x in range(self.winsize[0]):
                        gx = iu[0] + x
                        gy = iu[1] + y

                        # Brightness difference between next image and previous
                        bnext = nextImg[gy, gx]  # Brightness in next image
                        brightness_diff = bnext - bprev[idx]

                        # Compute AtWbnbp
                        AtWbnbp[0] += bw[idx % 4] * A[idx, 0] * brightness_diff  # Gradient in x
                        AtWbnbp[1] += bw[idx % 4] * A[idx, 1] * brightness_diff  # Gradient in y

                        idx += 1

                # Solve the linear system AtWA * deltaU = -AtWbnbp
                deltaU = np.matmul(invAtWA, -AtWbnbp)

                # Update u with deltaU
                u += deltaU

                # Early termination condition
                if np.linalg.norm(deltaU) < 1e-3:  # If deltaU is small, terminate early
                    break

            nextPts[ptidx] = u + halfWin

        return nextPts, status
