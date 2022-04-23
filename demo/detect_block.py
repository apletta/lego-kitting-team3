from BlockLoc import *
import cv2
import numpy as np

azure_kinect_rgb_image = cv.imread('one_red_one_blue.png')


def draw_detections(frame, blocks_red, blocks_blue):
    RAD_TO_DEG = 180 / np.pi

    for block in blocks_red:
        rect = ((block.x, block.y), (block.length, block.width),
                block.angle * RAD_TO_DEG)
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

    for block in blocks_blue:
        rect = ((block.x, block.y), (block.length, block.width),
                block.angle * RAD_TO_DEG)
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
# cv2.cvtColor(azure_kinect_rgb_image[200:850, 500:1300], cv2.
frame = azure_kinect_rgb_image#[200:850, 500:1300, :3]

blocks_red, blocks_blue = get_block_locs(frame)

draw_detections(frame, blocks_red, blocks_blue)

cv2.imshow('Webcam', frame)
cv2.waitKey(0)
