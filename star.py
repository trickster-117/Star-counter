import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

# install numpy with: pip install numpy
# install cv2 with: pip install opencv-python
# install scipy with: pip install scipy

# 1) read image
img = cv2.imread("hyades.jpg")

# 2) convert to grayscale. Colour are not needed
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3) Binarize image with a threshold, all pixels with a value over threshold
#    will be set to max_val, all others to zero
#    results in binarized image of ones and zeros
#    The lower the threshold, the more stars
max_val = 1
threshold = 220
_, img_bin = cv2.threshold(img_gray, threshold, max_val, cv2.THRESH_BINARY)

# 4) The actual smart function.
#    Assigns each connected area of pixels with non-zero values a label
#    ranging from 0 (background) to number of stars (1 - number of stars)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
labeled, nr_objects = ndimage.label(img_bin)
# Print number of stars
print("Number of stars is", nr_objects)

# 5) Find geometric center of each label (star with a certain area)
#    this corresponds to the middle of the star

# The coordinates of the stars will be gathered here
stars = []
# iterate through labels
for label in range(1, nr_objects):
    indices = np.argwhere(labeled == label)
    y_indices = indices[:, 0]
    x_indices = indices[:, 1]

    # geometric center
    iy_mean = int(np.mean(y_indices))
    ix_mean = int(np.mean(x_indices))

    stars.append([ix_mean, iy_mean])

# Result: Coordinates of stars in a 2D array
stars = np.array(stars)

# VISUALIZATION
# Shows the raw image
plt.figure(1)
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Raw image")

# Shows the gray scaled image
plt.subplot(2, 2, 2)
plt.imshow(img_gray, cmap="gray")
plt.title("Gray-scaled image")

# Shows the binarized image
plt.subplot(2, 2, 3)
# Attention: not all stars are shown due to visualization problems of python
# Therefore, the stars in the binarized image are enlarged to make them visible in the plot
kernel = np.ones((2, 2), np.uint8)
img_bin_vis = cv2.dilate(img_bin, kernel)
plt.imshow(img_bin_vis, cmap="gray")
plt.title("Binarised image")

# Shows the labels of the stars decoded as a colour
plt.subplot(2, 2, 4)
plt.imshow(labeled, cmap="gist_rainbow")
plt.title("Each star is decoded with a colour/label from 1 to number of stars.")

# Show the coordinates of the stars marked with a blue circle
# More stars can be found if the threshold is lowered.
plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Raw image")
plt.subplot(1, 2, 2)
plt.scatter(stars[:, 0], stars[:, 1], s=10, facecolors='none', edgecolors='blue')
plt.imshow(img, cmap="gray")
plt.title("marked coordinates of stars")

# Triggers the visualization
plt.show()
