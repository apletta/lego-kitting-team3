import numpy as np
import cv2

# device ID is 0 (default) for iam-camilo
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    cv2.imshow('Webcam',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()