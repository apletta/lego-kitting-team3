import numpy as np
import cv2
from BlockLoc import *

# device ID is 0 (default) for iam-camilo
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow('Webcam',frame)
cap.release()

P = np.asarray([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]])

blocks_red, blocks_blue = get_block_locs(frame)

print("blocks_red")
print('')
print(blocks_red)
print('')
print("blocks_blue")
print('')
print(blocks_blue)
