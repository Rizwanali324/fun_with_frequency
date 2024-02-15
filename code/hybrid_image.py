"""def hybrid_image(im1, im2, cutoff_low, cutoff_high):
    # you supply this code
    pass
"""

import numpy as np
import cv2

def hybrid_image(im1, im2, cutoff_low, cutoff_high):
    """
    Create a hybrid image by combining the low-frequency components of im1
    with the high-frequency components of im2.

    Parameters:
    - im1: The first image, from which low-frequency components are extracted.
    - im2: The second image, from which high-frequency components are extracted.
    - cutoff_low: The cutoff frequency for the low-pass filter applied to im1.
    - cutoff_high: The cutoff frequency for the high-pass filter applied to im2.

    Returns:
    - The hybrid image.
    """

    # Apply a low-pass filter to im1 to get its low-frequency components.
    # The cutoff_low determines the sigma for Gaussian blurring.
    low_pass = cv2.GaussianBlur(im1, (0, 0), sigmaX=cutoff_low, sigmaY=cutoff_low)

    # Apply a high-pass filter to im2 to get its high-frequency components.
    # This is done by subtracting a blurred version of im2 from the original im2.
    # The cutoff_high determines the sigma for Gaussian blurring.
    blurred_im2 = cv2.GaussianBlur(im2, (0, 0), sigmaX=cutoff_high, sigmaY=cutoff_high)
    high_pass = im2 - blurred_im2

    # Combine the low-frequency and high-frequency components.
    hybrid_image = np.clip(low_pass + high_pass, 0, 255)

    return hybrid_image.astype(np.uint8)



