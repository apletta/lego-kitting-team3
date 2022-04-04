import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Block import *

def view_axes(img, blocks):
    # img: BGR image
    # blocks: list of blocks
    plt.imshow(img[:,:,::-1])
    for block in blocks_blue:
        print("{} block: ({},{}); {} x {}; {} rad, {} deg".format(block.color, block.x, block.y, block.length, block.width, block.angle, block.angle*180/np.pi))
        len = 15
        dx1,dx2 = len*np.cos(block.angle), len*np.cos(block.angle-np.pi/2)
        dy1,dy2 = len*np.sin(block.angle), len*np.sin(block.angle-np.pi/2)
        plt.arrow(block.x,block.y,dx1,dy1,color='green')
        plt.arrow(block.x,block.y,dx2,dy2,color='pink')
    plt.show()

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

blocks_blue = []

for i in range(len(contours_blue)):
    # Convert contour into minimum area rectangle
    rect = cv.minAreaRect(contours_blue[i])
    # Convert minimum area rectangle into four points
    pts = cv.boxPoints(rect)
    # Compute length of two edges connecting to first point
    len1 = np.linalg.norm(pts[1]-pts[0])
    len2 = np.linalg.norm(pts[3]-pts[0])
    # Use longer edge to determine angle
    center = np.mean(pts,axis=0)
    if(len1 > len2):
        ang = np.arctan2(pts[1,1]-pts[0,1], pts[1,0]-pts[0,0])
        length = len1
        width = len2
    else:
        ang = np.arctan2(pts[3,1]-pts[0,1], pts[3,0]-pts[0,0])
        length = len2
        width = len1

    # TODO: GET PROJECTION MATRIX AND CONVERT TO X,Y IN ROBOT FRAME

    # TODO: fgure out angle of ang from minAreaRect

    # TODO: MAKE SURE THAT LENGTH AND WIDTH ARE ASSIGNED PROPERLY WITHIN CONSTRUCTOR

    cur_block = Block(center[0], center[1], length, width, ang,'blue')
    blocks_blue.append(cur_block)

# sort based on x coordinate of centroid
blocks_blue.sort(key=lambda cur: cur.x)
view_axes(img, blocks_blue)

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



