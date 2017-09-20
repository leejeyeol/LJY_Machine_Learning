import cv2
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize

# List for storing the LBP Histograms, address of images and the corresponding label
X_test = []
X_name = []
y_test = []

# For each image in the training set calculate the LBP histogram
# and update X_test, X_name and y_test
for train_image in train_images:
    # Read the image
    im = cv2.imread(train_image)
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
    # Number of points to be considered as neighbourers
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')

    '''
    # Calculate the histogram
    
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1] / sum(x[:, 1])
    # Append image path in X_name
    X_name.append(train_image)
    # Append histogram to X_name
    X_test.append(hist)
    # Append class label in y_test
    '''
# Display the training images
# when use it, calc histogram per superpixel.
