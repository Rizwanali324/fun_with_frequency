import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from skimage import io
import os

# Function to read points from file
def read_points(file_path):
    with open(file_path, 'r') as file:
        points = [tuple(map(float, line.split())) for line in file]
    return np.array(points)


# Ensure the folder exists
output_folder = 'tp3/tran_com_vis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to display visualization with triangulation and save on 's' key press
def display_and_save_on_key_press(image_path, points_path, output_path):
    # Load the image
    image = io.imread(image_path)
    
    # Read points from the file
    points = read_points(points_path)
    
    # Perform Delaunay triangulation
    tri = Delaunay(points)
    
    # Create figure for plotting
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), color='cyan')  # Triangulation
    ax.plot(points[:, 0], points[:, 1], 'o', color='red')  # Points
    plt.axis('off')  # Hide axis

    # Define the event handler for the 's' key press
    def on_key(event):
        if event.key == 's':
            plt.savefig(os.path.join(output_folder, output_path), bbox_inches='tight', pad_inches=0)
            print(f"Image saved to {os.path.join(output_folder, output_path)}")
            plt.close(fig)  # Optional: Close the figure after saving
    
    # Connect the event handler to the figure
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Display the figure
    plt.show()




# Example usage
image_path = "tp3/images/haba_mathieu.jpg"
points_path = "tp3/points/haba_mathieu.txt"
output_image_name = "triangulated_haba.jpg"

display_and_save_on_key_press(image_path, points_path, output_image_name)
(image_path, points_path, output_image_name)
