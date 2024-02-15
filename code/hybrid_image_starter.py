"""from imageio import imread
from align_images import align_images
from crop_image import crop_image
from hybrid_image import hybrid_image
from stacks import stacks
import matplotlib.pyplot as plt 

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift

# read images
im1 = imread('hybrid_python/Albert_Einstein.png', pilmode='L')
im2 = imread('hybrid_python/Marilyn_Monroe.png', pilmode='L')

# use this if you want to align the two images (e.g., by the eyes) and crop
# them to be of same size
im1, im2 = align_images(im1, im2)

# Choose the cutoff frequencies and compute the hybrid image (you supply
# this code)
arbitrary_value_1 = 5
arbitrary_value_2 = 2
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2
im12 = hybrid_image(im1, im2, cutoff_low, cutoff_high)

# Crop resulting image (optional)
assert im12 is not None, "im12 is empty, implement hybrid_image!"
im12 = crop_image(im12)

# Compute and display Gaussian and Laplacian Stacks (you supply this code)
n = 5  # number of pyramid levels (you may use more or fewer, as needed)
stacks(im12, n)


# After calling the hybrid_image function
plt.imshow(im12, cmap='gray')
plt.title('Hybrid Image')
plt.axis('off')
plt.show()


def low_pass_filter(image, sigma):
    #Apply a low-pass filter using a Gaussian blur.
    return gaussian_filter(image, sigma=sigma)

def high_pass_filter(image, sigma):
    #Apply a high-pass filter by subtracting the low-pass filtered image from the original
    return image - gaussian_filter(image, sigma=sigma)

def create_hybrid_image(image1, image2, sigma1, sigma2):
    #Create a hybrid image by combining a low-pass filtered image with a high-pass filtered one.
    low_pass = low_pass_filter(image1, sigma1)
    high_pass = high_pass_filter(image2, sigma2)
    return (low_pass + high_pass) / 2

def plot_frequency_analysis(image, title):
    #Plot the log magnitude of the Fourier transform of an image
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example usage with your previously loaded images (im1 and im2) and arbitrary sigma values
sigma1 = 19  # For the low-pass filter on the first image
sigma2 = 9 # For the high-pass filter on the second image

# Create the hybrid image
hybrid_image = create_hybrid_image(im1, im2, sigma1, sigma2)

# Display the hybrid image
plt.imshow(hybrid_image, cmap='gray')
plt.title('Hybrid Image')
plt.axis('off')
plt.show()

# Frequency analysis of the original, filtered, and hybrid images
plot_frequency_analysis(im1, 'Original Image 1 Frequency Analysis')
plot_frequency_analysis(im2, 'Original Image 2 Frequency Analysis')
plot_frequency_analysis(low_pass_filter(im1, sigma1), 'Low-Pass Filtered Image 1')
plot_frequency_analysis(high_pass_filter(im2, sigma2), 'High-Pass Filtered Image 2')
plot_frequency_analysis(hybrid_image, 'Hybrid Image Frequency Analysis')
"""


import os
import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.filters
import skimage.transform
import skimage.color
import matplotlib.pyplot as plt

# Check if an image is grayscale
def is_grayscale(image):
    return image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)

# Apply a Gaussian filter (lowpass) to the image
def apply_gaussian_filter(image, sigma):
    if image.ndim == 3:  # Image has multiple channels
        return np.stack([sk.filters.gaussian(channel, sigma=sigma) for channel in image.transpose((2, 0, 1))], axis=2)
    else:  # Grayscale image
        return sk.filters.gaussian(image, sigma=sigma)

# Apply a highpass filter to the image
def apply_highpass_filter(image, sigma):
    blurred = apply_gaussian_filter(image, sigma)
    highpass = image - blurred
    return highpass

# Create a hybrid image from two images
def create_hybrid_image(image1, image2, lowpass_cutoff, highpass_cutoff):
    lowpass_image1 = apply_gaussian_filter(image1, lowpass_cutoff)
    image2_resized = skimage.transform.resize(image2, lowpass_image1.shape[:2], mode='reflect', anti_aliasing=True)
    highpass_image2 = apply_highpass_filter(image2_resized, highpass_cutoff)
    hybrid_image = lowpass_image1 + highpass_image2
    return np.clip(hybrid_image, 0, 1), lowpass_image1, highpass_image2

# Plot and save the frequency analysis of an image
def plot_frequency_analysis(image, title, folder_path):
    if image.ndim == 3:
        image_gray = skimage.color.rgb2gray(image)
    else:
        image_gray = image
    fft_image = np.fft.fft2(image_gray)
    fft_shift = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f'Frequency Spectrum - {title}')
    plt.axis('off')
    plt.show()

    # Save the frequency analysis image
    save_image(magnitude_spectrum, folder_path, title.replace(" ", "_") + "_Frequency", cmap='gray')

# Save an image to a specified folder
def save_image(image, folder_path, title, cmap=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{title}.png")
    if cmap:
        plt.imsave(file_path, image, cmap=cmap)
    else:
        if image.dtype != np.uint8:
            image_to_save = (image * 255).astype(np.uint8)
        else:
            image_to_save = image
        skio.imsave(file_path, image_to_save)
    print(f"Image saved: {file_path}")

# Main execution starts here
folder_path = 'web/images/'
result_folder_path = 'web/results/hybrid/custom'

# Load imagesq
image1 = skio.imread(os.path.join(folder_path, 'biere.jpg'))
image2 = skio.imread(os.path.join(folder_path, 'eau.jpg'))

# Parameters for filtersqq
lowpass_cutoff = 30
highpass_cutoff = 10

# Create hybrid image and get lowpass and highpass images
hybrid_image, lowpass_image1, highpass_image2 = create_hybrid_image(image1, image2, lowpass_cutoff, highpass_cutoff)

# Display, save, and perform frequency analysis on each image
for img, title in zip([image1, lowpass_image1, image2, highpass_image2, hybrid_image], 
                      ["Image 1", "Lowpass Image 1", "Image 2", "Highpass Image 2", "Hybrid Image"]):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray' if is_grayscale(img) else None)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    # Save the displayed image

    save_image(img, result_folder_path, title.replace(" ", "_"))

    # Plot and save the frequency analysis for each image
    plot_frequency_analysis(img, title, result_folder_path)


