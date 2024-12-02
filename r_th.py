import scripts.BrukerMRI as bruker
import scripts.draw_rois as draw_rois
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class CineImageSelector:
    def __init__(self, cine_imgs):
        self.cine_imgs = cine_imgs
        self.num_imgs = cine_imgs.shape[2]
        self.current_image_index = 0
        self.selected_index = None

        # Set up the figure and axes
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)

        # Initial image display
        self.update_image()

        # Buttons for navigation and selection
        self.axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
        self.axnext = plt.axes([0.21, 0.05, 0.1, 0.075])
        self.axselect = plt.axes([0.6, 0.05, 0.2, 0.075])

        self.btn_next = Button(self.axnext, 'Next')
        self.btn_prev = Button(self.axprev, 'Previous')
        self.btn_select = Button(self.axselect, 'Select')

        self.btn_next.on_clicked(self.next_image)
        self.btn_prev.on_clicked(self.previous_image)
        self.btn_select.on_clicked(self.select_image)

    def update_image(self):
        """Updates the displayed image."""
        self.ax.clear()
        self.ax.imshow(self.cine_imgs[:, :, self.current_image_index], cmap='gray', origin='lower')
        self.ax.set_title(f"Image {self.current_image_index + 1} / {self.num_imgs}", fontsize=14)
        self.ax.axis('off')
        plt.draw()

    def next_image(self, event):
        """Displays the next image in the sequence."""
        self.current_image_index = (self.current_image_index + 1) % self.num_imgs
        self.update_image()

    def previous_image(self, event):
        """Displays the previous image in the sequence."""
        self.current_image_index = (self.current_image_index - 1) % self.num_imgs
        self.update_image()

    def select_image(self, event):
        """Saves the current image as the selected end-diastolic image."""
        self.selected_index = self.current_image_index
        plt.close(self.fig)

    def get_selected_index(self):
        """Returns the selected index."""
        return self.selected_index

#-------------Variables to set--------------#
# Primary data directory and experiment number(s) with labels
main_dir = '/Users/jonah/Documents/MRI_Data/Berkeley/HCM_Full/'
animal_id = '20241011_141940_M1913_1_1'
directory = main_dir + animal_id
exp_cine = 8
num_rays = 60

## Load cine data
cine_data = bruker.ReadExperiment(directory, exp_cine)
cine_imgs = cine_data.proc_data
cine_imgs = np.rot90(cine_imgs, k=2)

## Create the image selector and display the UI
selector = CineImageSelector(cine_imgs)

## Wait for the plot window to close
plt.show(block=True)

## After the plot is closed, the selected index is available
selected_index = selector.get_selected_index()
if selected_index is not None:
    print(f"Final selection: {selected_index + 1}")
else:
    print("No image was selected.")
    
## Set image and draw ROIs
end_diastole = cine_imgs[:,:,selected_index]
end_diastole = np.rot90(end_diastole, k=2)
end_diastole = np.fliplr(end_diastole)

mask = draw_rois.Rois_Avg(end_diastole)

def project_rays(mask, centroid, num_rays):
    """
    Projects rays from the centroid and calculates distances to the endocardium and epicardium.
    Args:
        mask (np.ndarray): Boolean mask where True = inside chamber, False = background.
        centroid (tuple): (x, y) coordinates of the centroid.
        num_rays (int): Number of rays to project.
    Returns:
        radii (list): Distances from centroid to endocardium.
        thicknesses (list): Distances from endocardium to epicardium.
    """
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    radii = []
    thicknesses = []

    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Traverse the ray pixel by pixel
        x, y = centroid
        ray_radius = None
        ray_thickness = None

        while 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1]:
            if mask[int(y), int(x)] and ray_radius is None:
                # First encounter with True (endocardium)
                ray_radius = np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2)
            if not mask[int(y), int(x)] and ray_radius is not None:
                # First encounter with False after endocardium (epicardium)
                ray_thickness = np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2) - ray_radius
                break
            x += dx
            y += dy

        if ray_radius is not None and ray_thickness is not None:
            radii.append(ray_radius)
            thicknesses.append(ray_thickness)

    return radii, thicknesses

def calculate_mean_ratio(radii, thicknesses):
    """
    Calculates the mean radius-to-thickness ratio.
    Args:
        radii (list): List of radii to endocardium.
        thicknesses (list): List of thicknesses from endocardium to epicardium.
    Returns:
        float: Mean R/Th ratio.
    """
    ratios = [r / t for r, t in zip(radii, thicknesses) if t > 0]
    return np.mean(ratios) if ratios else None

# Calculate centroid
def calc_centroid(mask):
    y, x = np.nonzero(mask)
    return (int(x.mean()), int(y.mean()))

# Calculate radii and thicknesses
centroid = calc_centroid(mask)
radii, thicknesses = project_rays(mask, centroid, num_rays)

# Calculate the mean R/Th ratio
mean_ratio = calculate_mean_ratio(radii, thicknesses)

print(f"Mean R/Th ratio: {mean_ratio:.2f}")

# Visualization
y_indices, x_indices = np.where(mask)
margin = 20  
x_min, x_max = max(np.min(x_indices) - margin, 0), min(np.max(x_indices) + margin, mask.shape[1])
y_min, y_max = max(np.min(y_indices) - margin, 0), min(np.max(y_indices) + margin, mask.shape[0])

plt.figure(figsize=(8, 8))  # Ensure a square figure
plt.imshow(end_diastole[y_min:y_max, x_min:x_max], cmap="gray", origin="upper")  # Adjust 'origin' to match the flipped image

# Adjust the centroid to the zoomed-in coordinates
centroid_adjusted = (centroid[0] - x_min, centroid[1] - y_min)

# Plot rays to endocardium and epicardium
for r, t, angle in zip(radii, thicknesses, np.linspace(0, 2 * np.pi, len(radii), endpoint=False)):
    # Endocardium
    x_r = centroid[0] + r * np.cos(angle)
    y_r = centroid[1] + r * np.sin(angle)
    
    # Adjust coordinates for zoomed-in view
    x_r_adjusted = x_r - x_min
    y_r_adjusted = y_r - y_min

    # Epicardium
    x_t = centroid[0] + (r + t) * np.cos(angle)
    y_t = centroid[1] + (r + t) * np.sin(angle)
    
    # Adjust coordinates for zoomed-in view
    x_t_adjusted = x_t - x_min
    y_t_adjusted = y_t - y_min

    # Plot rays
    plt.plot([centroid_adjusted[0], x_r_adjusted], [centroid_adjusted[1], y_r_adjusted], color="blue", alpha=0.5, linewidth=2)
    plt.plot([x_r_adjusted, x_t_adjusted], [y_r_adjusted, y_t_adjusted], color="green", alpha=0.5, linewidth=2)

# Plot adjusted centroid on top of the rays
plt.scatter(centroid_adjusted[0], centroid_adjusted[1], color="red", label="Centroid", zorder=5, s=100)

# Add legend and clean up
plt.legend(fontsize=12)
plt.axis("off")
plt.show()

