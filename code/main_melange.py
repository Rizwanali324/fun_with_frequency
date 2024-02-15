
 
import cv2
import numpy as np
import os




# Function to generate Gaussian and Laplacian pyramids for an image
def generate_pyramids(img, levels=6):
    # Gaussian Pyramid
    gaussian_pyramid = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        gaussian_pyramid.append(img)
    
    # Laplacian Pyramid
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(levels, 0, -1):
        size = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    
    return gaussian_pyramid, laplacian_pyramid[::-1]  # Return the Laplacian pyramid in correct order

# Function to blend two Laplacian pyramids
def blend_laplacian_pyramids(lap_pyramid1, lap_pyramid2):
    blended_pyramid = []
    for lap1, lap2 in zip(lap_pyramid1, lap_pyramid2):
        rows, cols, dpt = lap1.shape
        laplacian = np.hstack((lap1[:, 0:int(cols/2)], lap2[:, int(cols/2):]))
        blended_pyramid.append(laplacian)
    return blended_pyramid

# Function to reconstruct an image from its Laplacian pyramid
def reconstruct_from_pyramid(pyramid):
    reconstructed_image = pyramid[-1]  # Start with the smallest image
    for i in range(len(pyramid) - 2, -1, -1):  # Iterate upwards through the pyramid
        # Use the size of the next level as the destination size for pyrUp
        dstsize = (pyramid[i].shape[1], pyramid[i].shape[0])
        reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=dstsize)
        reconstructed_image = cv2.add(pyramid[i], reconstructed_image)
    return reconstructed_image



"""def display_or_save_pyramid(pyramid, title_prefix, save_dir=None):
    for i, level in enumerate(pyramid):
        window_title = f"{title_prefix} Pyramid Level {i}"
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, f"{window_title}.jpg"), level)
        else:
            cv2.imshow(window_title, level)
            cv2.waitKey(0)  # Wait for a key press to move to the next level
"""
def display_or_save_pyramid(pyramid, title_prefix, save_dir=None):
    for i, level in enumerate(pyramid):
        # Convert level to a proper format for visualization (float32 to uint8)
        normalized_level = cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX)
        vis_level = normalized_level.astype(np.uint8)
        
        # Apply histogram equalization to enhance visibility
        # Since the Laplacian pyramid levels can be negative and cv2.equalizeHist expects uint8,
        # we've already converted vis_level to uint8 above.
        # Note: Histogram equalization is applied to each channel separately in case of a color image.
        if len(vis_level.shape) == 2:  # Grayscale image
            vis_level = cv2.equalizeHist(vis_level)
        else:  # Color image, apply equalization to each channel
            channels = cv2.split(vis_level)
            eq_channels = [cv2.equalizeHist(ch) for ch in channels]
            vis_level = cv2.merge(eq_channels)

        window_title = f"{title_prefix} Pyramid Level {i}"
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, f"{window_title}.jpg"), vis_level)
        else:
            cv2.imshow(window_title, vis_level)
            cv2.waitKey(0)  # Wait for a key press to move to the next level







# Ensure the save directory exists, create if it doesn't
save_dir = "web/results/melange/custom_blinded"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)




# Load and resize images
img1 = cv2.imread("web/images/biere.jpg")
img1 = cv2.resize(img1, (500, 500))
img2 = cv2.imread("web\images\eau.jpg")
img2 = cv2.resize(img2, (500, 500))




# Generate pyramids for both images
_, laplacian_pyramid1 = generate_pyramids(img1)
_, laplacian_pyramid2 = generate_pyramids(img2)

# Blend the Laplacian pyramids
blended_pyramid = blend_laplacian_pyramids(laplacian_pyramid1, laplacian_pyramid2)

# Reconstruct the blended image from the blended pyramid
reconstructed = reconstruct_from_pyramid(blended_pyramid)
reconstructed=cv2.resize(reconstructed,(500,500))

def apply_laplacian_stack(img, kernels=[1, 3, 5, 7, 9, 11], save_dir=None, title_prefix="", intensity_scale=1.0, base_scale=0.5):
    """
    Args:
        base_scale: Factor to scale the original image before adding it back to the Laplacian result.
                    0 means no original image is added back; 1 means the original image is fully added back.
    """
    filenames = []  # List to store filenames of the saved images
    for k in kernels:
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=k)
        laplacian_scaled = laplacian * intensity_scale

        # Optionally add back a scaled version of the original image to preserve overall visibility
        base_image_scaled = img.astype(np.float32) * base_scale
        combined = laplacian_scaled + base_image_scaled

        # Normalize and convert to uint8 for visualization
        combined_vis = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if save_dir is not None:
            filename = f"{title_prefix}_Laplacian_K{k}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, combined_vis)
            filenames.append(save_path)

    return filenames

def display_saved_images(filenames):
    """Display saved images by their filenames."""
    for filename in filenames:
        img = cv2.imread(filename)
        if img is not None:
            cv2.imshow(filename, img)
        else:
            print(f"Failed to load image: {filename}")

# Apply the Laplacian stack function to the blended image, save the results, and get filenames
# Apply the Laplacian stack function to the blended image with customized parameters
saved_filenames = apply_laplacian_stack(
    img=reconstructed, 
    save_dir=save_dir, 
    title_prefix="enhanced_stack_blended_image",
    intensity_scale=1.0,  # Adjust as needed to modify the intensity of the Laplacian filter output
    base_scale=0.5  # Adjust to control how much of the original image is added back
)



# Load, resize, and generate pyramids for images as per your code

# Assuming the mask is created and applied to the images
# Example for applying a simple mask, adjust according to your actual mask application
"""mask = np.zeros_like(img1, dtype=np.uint8)
mask[:, :250] = 255  # Simple left-half mask for illustration
masked_img1 = cv2.bitwise_and(img1, mask)
masked_img2 = cv2.bitwise_and(img2, cv2.bitwise_not(mask))"""

mask = np.zeros_like(img1, dtype=np.uint8)
mask[:, :250] = 255  # Simple left-half mask for illustration
masked_img1 = cv2.bitwise_and(reconstructed, mask)
masked_img2 = cv2.bitwise_and(reconstructed, cv2.bitwise_not(mask))


# Display or save masked input images
cv2.imshow("Masked Image 1", masked_img1)
cv2.imshow("Masked Image 2", masked_img2)

# Apply the Laplacian stack function to the masked images, save the results, and get filenames
saved_filenames_masked_img1 = apply_laplacian_stack(masked_img1, save_dir=save_dir, title_prefix="stack_masked_image1")
saved_filenames_masked_img2 = apply_laplacian_stack(masked_img2, save_dir=save_dir, title_prefix="stack_masked_image2")

# Display the Laplacian stack images from saved files for the masked images
display_saved_images(saved_filenames_masked_img1)
display_saved_images(saved_filenames_masked_img2)



# Generate pyramids for both images
gaussian_pyramid1, laplacian_pyramid1 = generate_pyramids(masked_img1)
gaussian_pyramid2, laplacian_pyramid2 = generate_pyramids(masked_img2)


# Optionally display or save pyramid levels
display_or_save_pyramid(laplacian_pyramid1, "Image1 Laplacian", save_dir=save_dir)
display_or_save_pyramid(laplacian_pyramid2, "Image2 Laplacian", save_dir=save_dir)

# Display the original images and the blended result
cv2.imshow("image1", img1)
cv2.imshow("image2", img2)
cv2.imshow("Reconstructed", reconstructed)

# Display the Laplacian stack images from saved files
display_saved_images(saved_filenames)
# Save the blended image to the specified directory
save_path = os.path.join(save_dir, "blended_image_orple.jpg")



# Save the original images and the blended image to the specified directory

cv2.imwrite(os.path.join(save_dir, "Masked Image1.jpg" ),masked_img1)
cv2.imwrite(os.path.join(save_dir, "Masked Image2.jpg" ),masked_img2)
cv2.imwrite(os.path.join(save_dir, "original_image1.jpg"), img1)
cv2.imwrite(os.path.join(save_dir, "original_image2.jpg"), img2)
cv2.imwrite(os.path.join(save_dir, "blended_image_orple.jpg"), reconstructed)

print(f"Original and blended images saved to {save_dir}")

cv2.waitKey(0)
cv2.destroyAllWindows()





#_______________________________this code for irregular mask,

def apply_mask(img, mask):
    #Apply the given mask to the image.
    return cv2.bitwise_and(img, img, mask=mask)

# Load the images
img1_path = "web/images/vang1.jpg"
img2_path = "web/images/vang2.jpg"
mask_path = "web/images/vang_mask.jpg"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Ensure the images are the same size
img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))

# Load or create an irregular mask
# The mask should be the same size as the images, with white areas for img1, black for img2, and gray for blending
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask=cv2.resize(mask,(500,500))
mask_inv = cv2.bitwise_not(mask)

# Apply the mask to each image
img1_masked = apply_mask(img1, mask)
img2_masked = apply_mask(img2, mask_inv)


# Combine the masked images
blended = cv2.add(img1_masked, img2_masked)

# Save directory
save_dir = "web/results/melange/mask"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the original images, mask, and the blended image
cv2.imwrite(os.path.join(save_dir, "original_image1.jpg"), img1)
cv2.imwrite(os.path.join(save_dir, "original_image2.jpg"), img2)
cv2.imwrite(os.path.join(save_dir, "mask.jpg"), mask)
cv2.imwrite(os.path.join(save_dir, "blended_with_irregular_mask.jpg"), blended)

# Display the result
cv2.imshow("Original Image 1", img1)
cv2.imshow("Original Image 2", img2)
cv2.imshow("Mask", mask)
cv2.imshow("Blended with Irregular Mask", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
