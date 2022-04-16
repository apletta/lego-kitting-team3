import numpy as np
import cv2
from BlockLoc import *

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

# device ID is 0 (default) for iam-camilo
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    cv2.imshow('Webcam',frame)
    blocks_red, blocks_blue = get_block_locs(frame)

    view_axes(frame, blocks_red)
    view_axes(frame, blocks_blue)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()