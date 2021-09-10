import numpy as np
import cv2 as cv
import glob

# Read the image
img = cv.imread('/home/haahm/Development/projects/Results/Final_results_after_filter/testing_disparity/debug_00010.png')
print(img.shape[0])
#cv2.imshow('image',img)
height, width , channel = img.shape


# Cut the image in half
width_cutoff = width // 2
s1 = img[:, :width_cutoff]
s2 = img[:, width_cutoff:]

# Save each half
#np.save( '/home/haahm/Development/projects/Results/Final_results_after_filter/Divided_images/trained' + '.npy', s1)
cv.imwrite("/home/haahm/Development/projects/Results/Final_results_after_filter/Divided_images/after_spn"+ ".png", s1)
#np.save( '/home/haahm/Development/projects/Results/Final_results_after_filter/Divided_images/ground_truth' + '.npy', s2)
#cv.imwrite("/home/haahm/Development/projects/Results/Final_results_after_filter/Divided_images/ground_truth"+ ".png", s2)

