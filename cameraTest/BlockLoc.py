import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Block import *

def get_block_locs(img, P):
    """
    THIS FUNCTION TAKES AN IMAGE AND RETURNS BLOCK LOCATIONS AND ORIENTATIONS IN THE CAMERA FRAME, 
    THIS CURRENTLY WORKS FOR RED AND BLUE BLOCKS ONLY, IMAGE CROPPING IS HARD CODED


    INPUTS: img, P
    img: image, preferably cv.imread BGR
    P: projection matrix of camera, preferably a numpy array 3x4 of calibrated image

    OUTPUTS:
    blocks_red, blocks_blue
    blocks_red: sorted list of red blocks with attributes
    blocks_blue: sorted list of blue blocks with attributes    

    SEE Block.py for more info on attributes.
    """

    # crop image
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
    blue_mask = ((img_norm[:,:,0] > blue_thresh)*255).astype('uint8')
    red_mask = ((img_norm[:,:,2] > red_thresh)*255).astype('uint8')

    # set kernel for morphology stuff
    kernel = np.ones((3,3),np.uint8)

    # closes small patches
    blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_CLOSE, kernel)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)

    # open image 
    blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_OPEN, kernel)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)

    # find contours 
    contours_blue, _ = cv.findContours(image=blue_mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv.findContours(image=red_mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # initialize lists of red and blue blocks
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

        # GET PROJECTION MATRIX AND CONVERT TO X,Y in camera frame
        UVH = np.asarray([center[0],center[1],1])

        XYZH = np.linalg.pinv(P)@UVH

        X = XYZH[0]/XYZH[-1]
        Y = XYZH[1]/XYZH[-1]

        cur_block = Block(X,Y,length,width,ang,'blue')

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

        # GET PROJECTION MATRIX AND CONVERT TO X,Y IN camera frame 
        UVH = np.asarray([center[0],center[1],1])

        XYZH = np.linalg.pinv(P)@UVH

        X = XYZH[0]/XYZH[-1]
        Y = XYZH[1]/XYZH[-1]

        cur_block = Block(X,Y,length,width,ang,'red')

        blocks_red.append(cur_block)

    return blocks_red, blocks_blue
    