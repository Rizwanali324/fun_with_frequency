#    start code here 

import cv2
import numpy as np
import os

def build_gaussian_pyramid(img, levels, initial_sigma=1):
    gaussian_pyramid = [img]
    sigma = initial_sigma
    for level in range(1, levels):
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        img = cv2.pyrDown(blurred)
        gaussian_pyramid.append(img)
        sigma *= 2  # Double the sigma for the next level
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for level in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[level].shape[1], gaussian_pyramid[level].shape[0])
        upsampled = cv2.pyrUp(gaussian_pyramid[level + 1], dstsize=size)
        laplacian_level = cv2.subtract(gaussian_pyramid[level], upsampled)
        laplacian_pyramid.append(laplacian_level)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # The last level is added directly
    return laplacian_pyramid

def save_pyramid(pyramid, title, base_dir='web/results/pile/paramid1/custom'): # change only image original name for seprate folder(ex cat,eau)
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    for i, img in enumerate(pyramid):
        filename = f"{title}_Level_{i}.png"
        filepath = os.path.join(base_dir, filename)
        cv2.imwrite(filepath, img)
        print(f"Saved: {filepath}")

# Load an image (grayscale for simplicity)
img_path = 'web/results/hybrid/custom/Hybrid_Image.png'  # Change this to your image path
img = cv2.imread(img_path)

# Define the number of levels in the pyramid
levels = 8

# Build Gaussian and Laplacian pyramids
gaussian_pyramid = build_gaussian_pyramid(img, levels)
laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)

# Save the pyramids
save_pyramid(gaussian_pyramid, "Gaussian_Pyramid")
save_pyramid(laplacian_pyramid, "Laplacian_Pyramid")






def build_gaussian_stack(img, levels, initial_sigma=1):
    gaussian_stack = [img]
    sigma = initial_sigma
    for level in range(1, levels):
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        gaussian_stack.append(blurred)
        sigma *= 2  # Double the sigma for the next level
    return gaussian_stack

def build_laplacian_stack(gaussian_stack):
    laplacian_stack = []
    for level in range(len(gaussian_stack) - 1):
        laplacian_level = cv2.subtract(gaussian_stack[level], gaussian_stack[level + 1])
        laplacian_stack.append(laplacian_level)
    laplacian_stack.append(gaussian_stack[-1])  # The last level is added directly
    return laplacian_stack

def save_stack(stack, title, base_dir='web/results/pile/stack1/custom'): ## change the base dir for diff image(ex cat)
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    for i, img in enumerate(stack):
        filename = f"{title}_Level_{i}.png"
        filepath = os.path.join(base_dir, filename)
        cv2.imwrite(filepath, img)
        print(f"Saved: {filepath}")
"""
# Example usage
img_path = 'code/images/lincoln.jpg'  # Adjust this path as needed
img = cv2.imread(img_path)  # Load an image
levels = 8  # Define the number of levels in the stack
"""
# Build and save Gaussian and Laplacian stacks
gaussian_stack = build_gaussian_stack(img, levels)
laplacian_stack = build_laplacian_stack(gaussian_stack)
save_stack(gaussian_stack, "Gaussian")
save_stack(laplacian_stack, "Laplacian")