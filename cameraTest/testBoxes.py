import numpy as np
import cv2 as cv

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load image 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
img = cv.imread('Webcam_screenshot1_03.04.2022.png')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# roughly crop image 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO: fix crop box, and mark out workspace on board,
img = img[0:420,330:800,:]

# TODO: tune filter parameters on camera
lower_blue = np.array([70, 0, 0])
upper_blue = np.array([255, 100, 100])
mask_blue = cv.inRange(img, lower_blue, upper_blue)

# lower_red = np.array([100, 0, 0])
# upper_red = np.array([255, 100, 100])
# mask_red = cv.inRange(img, lower_red, upper_red)

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
cv.imshow('test',mask_blue)
# cv.imshow('test',imgKeyPoints)
cv.waitKey(0)
cv.destroyAllWindows()



# take locations u,v in image frame and convert these to the x,y,z frame for robot using projection matrix 
# create matrix of, row stucture: x,y,z,color 