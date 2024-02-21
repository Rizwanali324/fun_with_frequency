import numpy as np
from scipy.spatial import Delaunay
from skimage.transform import warp
from skimage import img_as_float
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import gaussian_filter
import cv2
import os
from trangular_compute import read_points

def interpolate_points(p1, p2, frac):
    """Interpolate between two sets of points."""
    return (1 - frac) * np.array(p1) + frac * np.array(p2)

def compute_affine_transform(ref_points, target_points):
    """Compute affine transformation matrix for a triangle."""
    ones = [1, 1, 1]
    A = np.vstack([ref_points.T, ones])
    B = np.vstack([target_points.T, ones])
    affine = np.dot(B, np.linalg.inv(A))
    return affine[:2, :]
"""
def apply_affine_transform(src, dst, src_tri, dst_tri, shape):
    transform = compute_affine_transform(src_tri, dst_tri)
    warp_coords = warp(dst, np.linalg.inv(transform), output_shape=shape)
    return np.multiply(src, warp_coords)"""



# Example usage (placeholders for the actual images and points)
# img1, img2: Images to morph
# img1_pts, img2_pts: Corresponding points in each image
# tri: Delaunay triangulation computed on the averaged points
# warp_frac, dissolve_frac: Parameters controlling the morph


"""
# Simplified affine transformation calculation
def calculate_affine_transform(src, dst):
    src_matrix = np.vstack([src.T, np.ones([1, src.shape[0]])])
    dst_matrix = np.vstack([dst.T, np.ones([1, dst.shape[0]])])
    affine_transform = np.linalg.lstsq(src_matrix.T, dst_matrix.T, rcond=None)[0]
    return affine_transform"""

# Function to warp each triangle and blend images
def warp_triangle(img, tri_coords, target_coords):
    # Create mask for triangle
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(target_coords), 1, 16, 0)
    mask = mask.astype(bool)
    
    # Calculate bounding box for efficiency
    bounding_rect = cv2.boundingRect(np.float32([target_coords]))
    (x, y, w, h) = bounding_rect
    cropped_mask = mask[y:y+h, x:x+w]
    
    # Calculate affine transform
    affine_transform = compute_affine_transform(tri_coords, target_coords - np.array([x, y]))
    
    # Apply affine transform
    warped_image = cv2.warpAffine(img[y:y+h, x:x+w], affine_transform[:2], (w, h), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    # Apply mask to warped image
    warped_image_masked = warped_image * cropped_mask[..., np.newaxis]
    return warped_image_masked, bounding_rect, cropped_mask

# Main morphing function
def morph(img1, img2, points1, points2, tri_indices, warp_frac, dissolve_frac):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    morphed_img = np.zeros(img1.shape, dtype=img1.dtype)

    for tri in tri_indices:
        # Get triangle coordinates for this triangle
        tri1 = points1[tri]
        tri2 = points2[tri]
        tri_target = (1.0 - warp_frac) * tri1 + warp_frac * tri2
        
        # Warp triangles from img1 and img2 to target
        warped_img1, bounding_rect, mask = warp_triangle(img1, tri1, tri_target)
        warped_img2, _, _ = warp_triangle(img2, tri2, tri_target)
        
        # Calculate weighted average (dissolve)
        x, y, w, h = bounding_rect
        morphed_img[y:y+h, x:x+w] += (1 - dissolve_frac) * warped_img1 + dissolve_frac * warped_img2
    
    return np.clip(morphed_img, 0, 255).astype(np.uint8)
def create_hybrid_image(img1, img2, sigma1, sigma2):
    # Apply low-pass filter to img1
    low_pass_img1 = gaussian_filter(img1, sigma=sigma1)
    
    # Apply high-pass filter to img2
    high_pass_img2 = img2 - gaussian_filter(img2, sigma=sigma2)
    
    # Combine the two images
    return low_pass_img1 + high_pass_img2
def add_border_points(points, image_shape):
    height, width = image_shape[0], image_shape[1]
    # Add corner points
    border_points = np.array([
        [0, 0], 
        [width - 1, 0], 
        [width - 1, height - 1], 
        [0, height - 1]
    ])
    # Optionally, add more points along the borders for denser coverage
    # For example, add midpoints of each edge
    midpoints = np.array([
        [width // 2, 0],
        [width - 1, height // 2],
        [width // 2, height - 1],
        [0, height // 2]
    ])
    # Combine original points with border and midpoint points
    all_points = np.vstack([points, border_points, midpoints])
    return all_points


def generate_frames(img1, img2, points1, points2, tri_indices, num_frames, output_dir):
    for i in range(num_frames):
        warp_frac = dissolve_frac = i / (num_frames - 1)
        morphed_img = morph(img1, img2, points1, points2, tri_indices, warp_frac, dissolve_frac)
        frame_filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        io.imsave(frame_filename, morphed_img)

def create_video_from_frames(frame_folder, output_video_file, fps=30):
    frame_files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".jpg")])
    if not frame_files:
        raise ValueError("No frames found in the directory")
    
    # Read the first frame to determine the video size
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' if you prefer
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)

    out.release()
# Adjust these variables as needed
num_frames = 100  # Number of frames for the animation
fps = 50  # Frames per second in the output video
output_frames_dir = 'tp3/video/morph_frames'  # Directory to save the frames
output_video_file = 'tp3/video/morph_animation.mp4'  # Path for the output video

# Make sure the frames output directory exists
if not os.path.exists(output_frames_dir):
    os.makedirs(output_frames_dir)
# Example usage
img1 = io.imread('tp3/images/Balde_Ismael.jpg')
img2 = io.imread('tp3/images/haba_mathieu.jpg')
# Read points and add border points
points1 = read_points('tp3/points/Balde_Ismael.txt')
points2 = read_points('tp3/points/haba_mathieu.txt')
points1 = add_border_points(points1, img1.shape)
points2 = add_border_points(points2, img2.shape)

# Compute average points for Delaunay triangulation
average_points = (points1 + points2) / 2
tri = Delaunay(average_points)
# Generate frames
generate_frames(img1, img2, points1, points2, tri.simplices, num_frames, output_frames_dir)

# Create video from frames
create_video_from_frames(output_frames_dir, output_video_file, fps=fps)

print(f"Animation video saved to {output_video_file}")

# Make sure the output directory exists
output_dir = 'tp3/others_results/hybrid_morphed'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



# Perform morphing
warp_frac = 0.5
dissolve_frac = 0.5
morphed_img = morph(img1, img2, points1, points2, tri.simplices, warp_frac, dissolve_frac)

# Create hybrid images from the original and morphed images
sigma1 = 15  # for low-pass filter
sigma2 = 5  # for high-pass filter
hybrid_original = create_hybrid_image(img1, img2, sigma1, sigma2)
hybrid_morphed = create_hybrid_image(morphed_img, morphed_img, sigma1, sigma2)

# Save the output images
io.imsave(os.path.join(output_dir, 'morphed_img.jpg'), morphed_img)
io.imsave(os.path.join(output_dir, 'hybrid_original.jpg'), hybrid_original)
io.imsave(os.path.join(output_dir, 'hybrid_morphed.jpg'), hybrid_morphed)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(morphed_img)
plt.title("Morphed Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(hybrid_original)
plt.title("Original Hybrid Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(hybrid_morphed)
plt.title("Morphed Hybrid Image")
plt.axis('off')

plt.show()