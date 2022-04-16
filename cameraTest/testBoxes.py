import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Block import *


# TODO:
# 1 CONVERT TO FUNCTION SUCH THAT WE CAN PROVIDE AN IMAGE AS INPUT, WITH THE STRUCT AS OUTPUT
# 2 FINISH UP TRANSFORMS
# 3 build out red after blue is done.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load image 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
img = cv.imread('Webcam_screenshot_03.04.2022.png')
# img = cv.imread('Webcam_screenshot1_03.04.2022.png')
# img = cv.imread('Webcam_screenshot2_03.04.2022.png')
# img = cv.imread('Webcam_screenshot3_03.04.2022.png')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# projection matrix of azure kinect 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
P = np.asarray([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]])

print(P)
print(np.shape(P))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# base_link to camera
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tf_camera_frame = np.asarray([
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,1]])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# roughly crop image 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO: fix crop box, and mark out workspace on board,
img = img[0:420,330:800,:] # BGR


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# color filter and create block masks 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Normalize image
img_mag = np.linalg.norm(img, axis=2)
img_norm = np.zeros(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_norm[i,j] = img[i,j] / img_mag[i,j]

# Thresholding and masking
blue_thresh = 0.8
red_thresh = 0.8
blue_mask = ((img_norm[:,:,0] > blue_thresh)*255).astype('uint8')
red_mask = ((img_norm[:,:,2] > red_thresh)*255).astype('uint8')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# clean up
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# set kernel 
kernel = np.ones((3,3),np.uint8)

# closes small patches
blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_CLOSE, kernel)
red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)

# open image 
blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_OPEN, kernel)
red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get contours
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
contours_blue, _ = cv.findContours(image=blue_mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

contours_red, _ = cv.findContours(image=red_mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

rects_blue = []

for i in range(len(contours_blue)):

    loc,dims,ang = cv.minAreaRect(contours_blue[i])

    # GET PROJECTION MATRIX AND CONVERT TO X,Y IN ROBOT FRAME
    UVH = np.asarray([loc[0],loc[1],1])

    XYZH = np.linalg.pinv(P)@UVH

    X = XYZH[0]/XYZH[-1]
    Y = XYZH[1]/XYZH[-1]

    # TODO: fgure out angle of ang from minAreaRect

    

    cur_block = Block(X,Y,dims[0],dims[1],ang,'blue')

    rects_blue.append(cur_block)

# sort based on x coordinate of centroid
rects_blue.sort(key=lambda cur: cur.x)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# image_copy = blue_mask.copy()
# image_copy = cv.cvtColor(image_copy,cv.COLOR_GRAY2BGR)
# cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
# # see the results
# cv.imshow('None approximation', image_copy)
# cv.waitKey(0)
# cv.imwrite('contours_none_image1.jpg', image_copy)
# cv.destroyAllWindows()

# cv.imshow('test',blue_mask)
# cv.imshow('test',imgKeyPoints)
# cv.waitKey(0)
# cv.destroyAllWindows()

# take locations u,v in image frame and convert these to the x,y,z frame for robot using projection matrix 
# create matrix of, row stucture: x,y,z,color 



