from astropy.modeling.models import Gaussian2D
import numpy as np
import matplotlib.pyplot as plt
from random import *
from projet_classe import MockImage
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from astropy.modeling.models import Gaussian2D
from skimage.draw import ellipse
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import Gaussian2D
from photutils.isophote import (Ellipse, EllipseGeometry,
                                build_ellipse_model)
#On créé l'image avec la classe
# Assuming your galaxy image is already in a numpy array called galaxy_image
image = MockImage()
image.show()
array = image.data
plt.show()






def process_array_with_custom_ellipse(array, center, minor_extreme1, minor_extreme2, major_extreme1, major_extreme2):
    mask = (array != 0)
    
    # Calculate major and minor axis lengths
    major_axis = np.linalg.norm(np.array(major_extreme1) - np.array(major_extreme2))
    minor_axis = np.linalg.norm(np.array(minor_extreme1) - np.array(minor_extreme2))
    
    # Calculate angle of rotation for major axis
    angle = np.arctan2(major_extreme1[0]-major_extreme2[0], major_extreme1[1]-major_extreme2[1])
    print(angle)
    
    # Generate coordinates for the ellipse perimeter
    rr, cc = ellipse(center[1], center[0], minor_axis / 2 + 30, major_axis / 2 + 30, rotation=(angle+np.pi/2))
    
    include_mask = np.zeros_like(array, dtype=bool)
    include_mask[rr, cc] = True

    valid_mask = mask & ~include_mask

    filtered_array = array[valid_mask]
    mean_value = np.mean(filtered_array)
    print(mean_value)
    # Subtract the mean value from the whole array
    processed_array = array - mean_value

    # Set values outside the ellipse to 0
    processed_array[valid_mask] = 0
    
    return processed_array



# Example usage
image_data = image.data # Replace this with your actual image data
center = (398, 508)
minor_extreme1 = (330, 570)
minor_extreme2 = (453, 468)
major_extreme1 = (600, 500)
major_extreme2 = (400, 300)

processed_image = process_array_with_custom_ellipse(image_data, center, minor_extreme1, minor_extreme2, major_extreme1, major_extreme2)

# Plot the processed image
plt.imshow(processed_image, cmap='gray', origin='lower')

# Calculate the linear function
x_values = [major_extreme1[1], major_extreme2[1]]
y_values = [major_extreme1[0], major_extreme2[0]]
coefficients = np.polyfit(x_values, y_values, 1)
linear_function = np.poly1d(coefficients)

# Plot the linear function
x_range = np.linspace(min(x_values), max(x_values), 100)
plt.plot(x_range, linear_function(x_range), color='red', label='Linear Function')

# Set plot labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show the plot
plt.show()






sum_non_zero = 0

# Iterate through the array
for row in processed_image:
    for value in row:
        if value != 0:
            sum_non_zero += value
pix = sum_non_zero
print(pix)
#zp = 24.897

#mab = -2.5*np.log10(pix) + zp

#den = (10**(mab/-2.5))* 3.631*10**-20

#flux = den * 1/(2.8111 * 10**-7 )

#Lum = 4*np.pi* flux * (3.79418*10**27 )**2
#print(Lum)







# Apply image segmentation to label regions
label_image = label(processed_image)

# Calculate region properties
regions = regionprops(label_image)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(label2rgb(label_image, image=processed_image), cmap=plt.cm.gray, origin='lower')

for region in regions:
    # Highlight galaxy regions with a certain area (you can adjust this threshold)
    if region.area > 100:  # Adjust the area threshold as needed
        minr, minc, maxr, maxc = region.bbox
        
_values = [major_extreme1[1], major_extreme2[1]]
y_values = [major_extreme1[0], major_extreme2[0]]
coefficients = np.polyfit(x_values, y_values, 1)
linear_function = np.poly1d(coefficients)

# Plot the linear function
x_range = np.linspace(min(x_values), max(x_values), 100)
plt.plot(x_range, linear_function(x_range), color='red', label='Linear Function')

# Set plot labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show the plot



plt.show()




# Assuming you have the 'image.data' array and other provided variables




x = image.data
data = x
center = (398, 508)
minor_extreme1 = (330, 570)
minor_extreme2 = (453, 468)
major_extreme1 = (600, 500)
major_extreme2 = (400, 300)
meanvalue = 5.004167955126319
major_axis = np.linalg.norm(np.array(major_extreme1) - np.array(major_extreme2))
minor_axis = np.linalg.norm(np.array(minor_extreme1) - np.array(minor_extreme2))
angle = np.arctan2(major_extreme1[1] - major_extreme2[1], major_extreme1[0] - major_extreme2[0])
# Index of the outermost isophote level


# Print the values
print(f"Semi-major axis value : {major_axis}")
print(f"Position angle value : {angle}")
print(f"Center y-coordinate, x-coordinate value : {center}")





geometry = EllipseGeometry(x0=center[0], y0=center[1], sma= 50, eps=0.6, pa=(angle + np.pi / 2))
ellipse = Ellipse(data, geometry)

isolist = ellipse.fit_image()

model_image = build_ellipse_model(data.shape, isolist)

residual = data - model_image
# Index of the outermost isophote level
outermost_index = -1  # Index of the last element

# Access properties for the outermost isophote
outermost_sma = isolist.sma[outermost_index]
outermost_eps = isolist.eps[outermost_index]
outermost_pa = isolist.pa[outermost_index]
outermost_x0 = isolist.x0[outermost_index]
outermost_y0 = isolist.y0[outermost_index]

# Print the values
print(f"Semi-major axis value of outermost isophote: {outermost_sma}")
print(f"Ellipticity value of outermost isophote: {outermost_eps}")
print(f"Position angle value of outermost isophote: {outermost_pa}")
print(f"Center x-coordinate value of outermost isophote: {outermost_x0}")
print(f"Center y-coordinate value of outermost isophote: {outermost_y0}")
plt.imshow(model_image, origin='lower')
plt.title('Model Image')
plt.colorbar()
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 5), nrows=1, ncols=3)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)

ax1.imshow(data, origin='lower', extent=(0, data.shape[1], 0, data.shape[0]))
ax1.set_title('Data')

smas = np.linspace(10, 50, 20)

for sma in smas:
    iso = isolist.get_closest(sma)
    x, y = iso.sampled_coordinates()
    ax1.plot(x, y, color='white')

ax2.imshow(model_image, origin='lower', extent=(0, model_image.shape[1], 0, model_image.shape[0]))
ax2.set_title('Ellipse Model')

ax3.imshow(residual, origin='lower', extent=(0, residual.shape[1], 0, residual.shape[0]))
ax3.set_title('Residual')
plt.show()
sum_non_zero = 0
x = model_image - meanvalue

x[x < 0] = 0
# Iterate through the array
for row in x:
    for value in row:
        if value != 0:
            sum_non_zero += value
pix = sum_non_zero
print(pix)