import math
import cv2
import mediapipe as mp
import time
import hand_tracking_module as htm
import numpy as np

cap = cv2.VideoCapture(0)

def masked_hand(lmlist):
    img1 = np.zeros((512, 512, 1), dtype="uint8")
    for i in range(21):
        x = lmlist[i][1]
        y = lmlist[i][2]
        cv2.circle(img1, (x, y), 5, (255, 255, 0), cv2.FILLED)
    return img1


ctime=0
stime=0

detector = htm.handDetector()
while True:
    success, img = cap.read()

    img = detector.findhands(img)
    lmlist = detector.findposition(img)
    img2 = np.zeros((512, 512, 1), dtype="uint8")
    if len(lmlist) != 0:
        img2 = masked_hand(lmlist)

    ctime=time.time()
    fps=1/(ctime-stime)
    stime=ctime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0), 3)

    cv2.imshow("Image", img2)
    cv2.waitKey(1)