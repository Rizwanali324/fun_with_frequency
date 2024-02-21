# Setup: Imports and Initial Definitions
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from skimage.transform import warp
import cv2
from scipy.spatial import distance_matrix
import os

# Function Definitions
def read_points(filename):
    """Reads control points from a file into a NumPy array."""
    with open(filename) as file:
        points = [tuple(map(float, line.split())) for line in file]
    return np.array(points)

def add_border_points(points, image_shape):
    """Augments control points with additional points around the image borders and midpoints."""
    height, width = image_shape[0], image_shape[1]
    border_points = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
    ])
    midpoints = np.array([
        [width // 2, 0], [width - 1, height // 2], 
        [width // 2, height - 1], [0, height // 2]
    ])
    all_points = np.vstack([points, border_points, midpoints])
    return all_points

# Core Function for RBF-based Image Morphing
def rbf_morph(source_image, target_image, source_points, target_points, alpha=0.5):
    """Morphs the source image towards the target image using RBF interpolation."""
    sp = np.array(source_points)
    tp = np.array(target_points)
    inter_points = (1 - alpha) * sp + alpha * tp
    sp_x, sp_y = sp[:, 0], sp[:, 1]
    tp_x, tp_y = inter_points[:, 0], inter_points[:, 1]
    rbf_x = Rbf(sp_x, sp_y, tp_x, function='thin_plate', smooth=0)
    rbf_y = Rbf(sp_x, sp_y, tp_y, function='thin_plate', smooth=0)

    def transform(coords):
        x, y = coords[:, 0], coords[:, 1]
        return np.vstack([rbf_x(x, y), rbf_y(x, y)]).T
    
    return warp(source_image, transform, output_shape=source_image.shape)

# Execution: Loading Images, Preparing Points, and Applying RBF-based Morphing
img1 = cv2.cvtColor(cv2.imread('tp3/images/Balde_Ismael.jpg'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('tp3/images/haba_mathieu.jpg'), cv2.COLOR_BGR2RGB)
points1 = read_points('tp3/points/Balde_Ismael.txt')
points2 = read_points('tp3/points/haba_mathieu.txt')
points1 = add_border_points(points1, img1.shape)
points2 = add_border_points(points2, img2.shape)





def apply_transform_vectorized(image, src_points, dst_points):
    """Applies a geometric transformation to the image based on control points using a vectorized approach."""
    # Create a grid of coordinates in the original image
    coords_y, coords_x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    coords_flat = np.stack([coords_x.flatten(), coords_y.flatten()], axis=-1)

    # Compute distances between each pixel and the control points
    dists = distance_matrix(coords_flat, src_points)

    # Compute weights for each control point for each pixel
    weights = 1.0 / np.maximum(dists, 1e-9)  # Prevent division by zero
    weights /= weights.sum(axis=1, keepdims=True)  # Normalize weights

    # Compute the weighted average of the control point shifts
    shifts = dst_points - src_points
    weighted_shifts = np.dot(weights, shifts)

    # Apply shifts to the original coordinates
    transformed_coords = coords_flat + weighted_shifts
    transformed_coords_x = transformed_coords[:, 0].reshape(image.shape[0], image.shape[1])
    transformed_coords_y = transformed_coords[:, 1].reshape(image.shape[0], image.shape[1])

    # Map the pixels of the original image to the new coordinates
    transformed_image = cv2.remap(image, transformed_coords_x.astype(np.float32), transformed_coords_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return transformed_image

# Applying Vectorized WLS Morphing
morphed_image_wls = apply_transform_vectorized(img1, points1, points2)

# Applying RBF-based Morphing
morphed_image_rbf = rbf_morph(img1, img2, points1, points2, alpha=0.5)

# Ensure the directory exists
output_dir = 'tp3/others_results/non_tri'
os.makedirs(output_dir, exist_ok=True)

# Displaying the Morphing Results in a Single Plot
plt.figure(figsize=(12, 6))

# RBF-based Morphing Result
plt.subplot(1, 2, 1)
plt.imshow(morphed_image_rbf)
plt.title('RBF-based Morphing')
plt.axis('off')

# Vectorized WLS Morphing Result
plt.subplot(1, 2, 2)
plt.imshow(morphed_image_wls)
plt.title('Vectorized WLS Morphing')
plt.axis('off')

# Save the figure
plt.savefig(os.path.join(output_dir, 'morphing_comparison.png'))

# Show the plot
plt.show()
