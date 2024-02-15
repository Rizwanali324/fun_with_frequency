




import cv2
import numpy as np
import os

def ensure_dir(file_path):
    """
    Ensure that a directory exists, and if not, create it.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_log_magnitude_fourier(image):
    """
    Compute the log magnitude of the Fourier Transform of a grayscale image.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(image_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return magnitude_spectrum

def make_hybrid_image_with_fourier_analysis(image1_path, image2_path, sigma1=5, sigma2=5, visualize=False, save_results=True):
    """
    Create a hybrid image from RGB images, perform frequency analysis, and optionally visualize and save the log magnitude
    of the Fourier Transform of the input, filtered, and hybrid images.
    """
    # Paths for saving results
    results_dir = 'web/results/hybrid/Einstein-Marilyn'
    ensure_dir(results_dir + '/')  # Ensure the directory exists
    
    # Read and resize images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Gaussian blur for low and high frequencies in RGB
    low_freq_image = cv2.GaussianBlur(image1, (0, 0), sigma1)
    high_freq_image = cv2.GaussianBlur(image2, (0, 0), sigma2)
    high_freq_image = image2 - high_freq_image

    # Hybrid image
    hybrid_image = np.clip(low_freq_image + high_freq_image, 0, 255).astype(np.uint8)

    # Fourier analysis in grayscale
    fourier_images = {
        "Image1_Fourier.jpg": compute_log_magnitude_fourier(image1),
        "Image2_Fourier.jpg": compute_log_magnitude_fourier(image2),
        "Low_Freq_Fourier.jpg": compute_log_magnitude_fourier(low_freq_image),
        "High_Freq_Fourier.jpg": compute_log_magnitude_fourier(high_freq_image),
        "Hybrid_Image_Fourier.jpg": compute_log_magnitude_fourier(hybrid_image)
    }

    if visualize or save_results:
        for filename, img in fourier_images.items():
            path = os.path.join(results_dir, filename)
            if visualize:
                cv2.imshow(filename, img / img.max())
            if save_results:
                cv2.imwrite(path, img)
        
        if save_results:
            cv2.imwrite(os.path.join(results_dir, 'Hybrid_Image_cat.jpg'), hybrid_image)
        
        if visualize:
            cv2.imshow('Hybrid Image (RGB)', hybrid_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return hybrid_image, fourier_images

# Example usage
image1_path = 'web/images/Marilyn_Monroe.png '
image2_path = 'web/images/Albert_Einstein.png'
hybrid_image, fourier_analysis = make_hybrid_image_with_fourier_analysis(image1_path, image2_path, sigma1=30, sigma2=5, visualize=True, save_results=True)

