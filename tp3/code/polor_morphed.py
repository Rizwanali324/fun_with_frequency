from skimage import io, transform, img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the output directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def to_polar_image(img, img_name, output_dir):
    center = (img.shape[0] // 2, img.shape[1] // 2)
    radius = np.sqrt((center[0])**2 + (center[1])**2)
    if img.ndim == 3:  # RGB image
        polar_image = transform.warp_polar(img, radius=radius, output_shape=img.shape[:2], center=center, channel_axis=2)
    else:  # Grayscale image
        polar_image = transform.warp_polar(img, radius=radius, output_shape=img.shape, center=center)
    polar_image_uint8 = img_as_ubyte(polar_image)
    polar_image_path = os.path.join(output_dir, f'{img_name}_polar.jpg')
    io.imsave(polar_image_path, polar_image_uint8)
    return polar_image

def morph_images(img1, img2, img_name, output_dir, alpha=0.5):
    assert img1.shape == img2.shape, "Images must have the same dimensions for morphing."
    morphed_img = (1 - alpha) * img1 + alpha * img2
    morphed_img_uint8 = img_as_ubyte(morphed_img)
    morphed_img_path = os.path.join(output_dir, f'{img_name}_morphed.jpg')
    io.imsave(morphed_img_path, morphed_img_uint8)
    return morphed_img

# Directory and image setup
output_dir = 'tp3/others_results/polar_trans/'
ensure_dir(output_dir)

image_path1 = 'tp3/images/Balde_Ismael.jpg'
image_path2 = 'tp3/images/haba_mathieu.jpg'

# Read and process images
img1 = img_as_float(io.imread(image_path1))
img2 = img_as_float(io.imread(image_path2))

img1_name = os.path.splitext(os.path.basename(image_path1))[0]
img2_name = os.path.splitext(os.path.basename(image_path2))[0]

img1_polar = to_polar_image(img1, img1_name, output_dir)
img2_polar = to_polar_image(img2, img2_name, output_dir)
img_morphed = morph_images(img1, img2, "combined", output_dir)

# Convert the morphed image into polar coordinates
img_morphed_polar = to_polar_image(img_morphed, "combined_morphed", output_dir)

# Displaying images
fig, axes = plt.subplots(4, 2, figsize=(20, 10))

# Setup for display titles and images in a loop
titles = ['Original Image 1', 'Original Image 2', 'Polar Image 1', 'Polar Image 2', 'Morphed Image', 'Polar Morphed Image']
images = [img1, img2, img1_polar, img2_polar, img_morphed, img_morphed_polar]

for i, (title, image) in enumerate(zip(titles, images)):
    row = i // 2
    col = i % 2
    axes[row, col].imshow(image, cmap='gray' if image.ndim == 2 else None)
    axes[row, col].set_title(title)
    axes[row, col].axis('off')

# Remove unused subplots if any
for j in range(i + 1, axes.size):
    axes.flat[j].axis('off')

plt.tight_layout()
plt.show()
