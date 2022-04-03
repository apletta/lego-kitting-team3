import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load image 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
img = cv.imread('Webcam_screenshot_03.04.2022.png')
# img = cv.imread('Webcam_screenshot1_03.04.2022.png')
# img = cv.imread('Webcam_screenshot2_03.04.2022.png')
# img = cv.imread('Webcam_screenshot3_03.04.2022.png')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# roughly crop image 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO: fix crop box, and mark out workspace on board,
img = img[0:420,330:800,:] # BGR

# Normalize image
img_mag = np.linalg.norm(img, axis=2)
img_norm = np.zeros(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_norm[i,j] = img[i,j] / img_mag[i,j]
# Thresholding and masking
blue_thresh = 0.8
red_thresh = 0.8
blue_mask = img_norm[:,:,0] > blue_thresh
red_mask = img_norm[:,:,2] > red_thresh

# TODO: EROSION NOISE FILTERING

# blob filtering for masked images, this returns locations in x/y for image 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# set parameters for blob detection 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# params = cv.SimpleBlobDetector_Params()

# # Change thresholds
# params.minThreshold = 1
# params.maxThreshold = 255

# # # Filter by Area.
# # params.filterByArea = True
# # params.minArea = 1500

# # # Filter by Circularity
# # params.filterByCircularity = True
# # params.minCircularity = 0.1

# # # Filter by Convexity
# # params.filterByConvexity = True
# # params.minConvexity = 0.87

# # # Filter by Inertia
# # params.filterByInertia = True
# # params.minInertiaRatio = 0.01


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # create detector
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# detector = cv.SimpleBlobDetector_create(params)
# detector = cv.SimpleBlobDetector_create()

# keypoints = detector.detect(img)

# print(keypoints)

# imgKeyPoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# cv.imshow('test',img)
# cv.imshow('test',imgKeyPoints)
cv.waitKey(0)
cv.destroyAllWindows()



# take locations u,v in image frame and convert these to the x,y,z frame for robot using projection matrix 
# create matrix of, row stucture: x,y,z,color 