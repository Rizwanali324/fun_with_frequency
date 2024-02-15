import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import os

def sharpen_image(image, sigma=2, alpha=1.5):
    blurred_image = gaussian_filter(image, sigma=sigma)
    mask = image - blurred_image
    sharpened_image = image + alpha * mask
    sharpened_image = np.clip(sharpened_image, 0, 255)
    return sharpened_image.astype(np.uint8)

def resize_image(image, height=300):
    """Resize the image to a given height while maintaining the aspect ratio."""
    ratio = height / image.shape[0]
    width = int(image.shape[1] * ratio)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image

# Initialize a list to hold the processed images globally
processed_images = []

def on_trackbar_change(dummy=None):
    global processed_images
    sigma = cv2.getTrackbarPos('Sigma', 'Images')
    alpha = cv2.getTrackbarPos('Alpha', 'Images') / 10.0

    processed_images = []

    # Process each image, sharpen, and resize for display
    for img, path in zip(original_imgs, image_paths):
        sharpened = sharpen_image(img, sigma=sigma, alpha=alpha)
        resized_original = resize_image(img)
        resized_sharpened = resize_image(sharpened)
        processed_images.append((np.hstack((resized_original, resized_sharpened)), path))

    # Concatenate all images horizontally
    display_img = np.hstack([img for img, _ in processed_images])
    
    # Show the concatenated images
    cv2.imshow('Images', display_img)

def save_images(processed_images):
    save_dir = 'web/results/accentuation'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img, path in processed_images:
        filename = os.path.join(save_dir, os.path.basename(path).replace('.png', '_sharpened.png'))
        cv2.imwrite(filename, img)
        print(f'Saved: {filename}')

if __name__ == '__main__':
    image_paths = ['web/images/Albert_Einstein.png', 'web/images/Marilyn_Monroe.png']
    original_imgs = [np.array(Image.open(path)) for path in image_paths]

    # Create a window
    cv2.namedWindow('Images')

    # Create trackbars for sigma and alpha
    cv2.createTrackbar('Sigma', 'Images', 0, 10, on_trackbar_change)
    cv2.createTrackbar('Alpha', 'Images', 10, 50, on_trackbar_change)

    # Initial call to update the display
    on_trackbar_change()

    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_images(processed_images)  # Save the currently displayed images

    cv2.destroyAllWindows()
