import numpy as np
import cv2

# device ID is 0 (default) for iam-camilo
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(img,[box],0,(0,0,255),2)

    cv2.imshow('Webcam',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()