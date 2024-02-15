"""def stacks(im12, n):
    # you supply this code
    pass
"""

import numpy as np
import cv2

def stacks(im12, n):
    """
    Generate a stack of n images transitioning from one perceptual component to another.

    Parameters:
    - im12: The hybrid image from which to generate the stack.
    - n: The number of images in the stack.

    Returns:
    - A list of n images, where each image is a step in the transition.
    """
    # Initialize an empty list to store the stack of images.
    stack = []

    # Determine the maximum amount of blurring to apply initially.
    # This is somewhat arbitrary and can be adjusted.
    max_blur_kernel_size = 31  # Using an odd number for kernel size; adjust as needed.

    # Generate the stack
    for i in range(n):
        # Calculate the kernel size for the current step's blurring effect.
        # Linearly interpolate between the max kernel size and 1 (no blurring).
        kernel_size = int(np.linspace(max_blur_kernel_size, 1, n)[i])
        if kernel_size % 2 == 0: kernel_size += 1  # Ensure kernel size is odd.

        # Apply Gaussian blur with the current kernel size.
        blurred = cv2.GaussianBlur(im12, (kernel_size, kernel_size), 0)

        # Add the blurred image to the stack.
        stack.append(blurred)

    return stacks
