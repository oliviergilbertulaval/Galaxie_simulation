from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_noise_image
import numpy as np
import matplotlib.pyplot as plt
from random import *
from projet_classe import MockImage
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb

#On créé l'image avec la classe
# Assuming your galaxy image is already in a numpy array called galaxy_image
image = MockImage(randomize=True, number_of_galaxies=10)
image.show()

plt.show()
#Utilise image.data pour accéder au tableau nu  mpy de tes données
#Ex:
def mean_and_median_with_threshold(array, max_threshold):
    mask = (array != 0) & (array <= max_threshold)
    filtered_array = array[mask]
    mean_value = np.mean(filtered_array)
    median_value = np.median(filtered_array)
    return median_value, mean_value
x = mean_and_median_with_threshold(image.data, 50)
print(x)
threshold = x[1]



print("Original Image Statistics:")
print("Max og:", np.max(image.data))
print("Min og:", np.min(image.data))
print("Mean og:", np.mean(image.data))
print("Median og:", np.median(image.data))
# Define the threshold value
#threshold = np.mean(image.data)

# Set values under the threshold to 0 while keeping dimensions
masked_array = np.where(image.data < threshold, 0, image.data)



#plt.imshow(masked_array, cmap='gray', origin='lower')
#plt.show()


def mean_and_median_with_threshold(array, max_threshold):
    mask = (array != 0) & (array <= max_threshold)
    filtered_array = array[mask]
    mean_value = np.mean(filtered_array)
    median_value = np.median(filtered_array)
    return median_value, mean_value

new_me_med = mean_and_median_with_threshold(masked_array , 25)
print(new_me_med)


x = np.where(masked_array < new_me_med[1], 0, masked_array)

plt.imshow(x, cmap='gray', origin='lower')
plt.show()

# Apply image segmentation to label regions
label_image = label(x)

# Calculate region properties
regions = regionprops(label_image)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(label2rgb(label_image, image=x), cmap=plt.cm.gray, origin='lower')

for region in regions:
    # Highlight galaxy regions with a certain area (you can adjust this threshold)
    if region.area > 100:  # Adjust the area threshold as needed
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

plt.show()


