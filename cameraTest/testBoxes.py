import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Block import *

def view_axes(img, blocks):
    # img: BGR image
    # blocks: list of blocks
    plt.imshow(img[:,:,::-1])
    for block in blocks:
        print("{} block: ({},{}); {} x {}; {} rad, {} deg".format(block.color, block.x, block.y, block.length, block.width, block.angle, block.angle*180/np.pi))
        len = 15
        dx1,dx2 = len*np.cos(block.angle), len*np.cos(block.angle-np.pi/2)
        dy1,dy2 = len*np.sin(block.angle), len*np.sin(block.angle-np.pi/2)
        plt.arrow(block.x,block.y,dx1,dy1,color='green')
        plt.arrow(block.x,block.y,dx2,dy2,color='pink')
    plt.show()

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

blocks_blue = []
blocks_red = []

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

    # GET PROJECTION MATRIX AND CONVERT TO X,Y IN ROBOT FRAME
    UVH = np.asarray([center[0],center[1],1])

    XYZH = np.linalg.pinv(P)@UVH

    X = XYZH[0]/XYZH[-1]
    Y = XYZH[1]/XYZH[-1]

    cur_block = Block(center[0],center[1],length,width,ang,'blue')

    blocks_blue.append(cur_block)

for i in range(len(contours_red)):
    # Convert contour into minimum area rectangle
    rect = cv.minAreaRect(contours_red[i])
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

    # GET PROJECTION MATRIX AND CONVERT TO X,Y IN ROBOT FRAME
    UVH = np.asarray([center[0],center[1],1])

    XYZH = np.linalg.pinv(P)@UVH

    X = XYZH[0]/XYZH[-1]
    Y = XYZH[1]/XYZH[-1]

    cur_block = Block(center[0],center[1],length,width,ang,'red')

    blocks_red.append(cur_block)

# sort based on x coordinate of centroid
# blocks_blue.sort(key=lambda cur: cur.x)
# blocks_red.sort(key=lambda cur: cur.x)
# view_axes(img, blocks_blue)
# view_axes(img, blocks_red)
