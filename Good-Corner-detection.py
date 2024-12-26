import cv2
import numpy as np

base = np.zeros((480, 640, 3), dtype=np.uint8)
base2 = base.copy()

cv2.rectangle(base, (50, 100), (590,400), (255,0,0), -1)
cv2.rectangle(base, (250, 40), (400, 100), (255,0,0), -1)

gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 50, useHarrisDetector=True, k=0.04, blockSize=2)

for corner in corners:
    x = int(corner[0][0])
    y = int(corner[0][1])

    cv2.circle(base2, (x,y), 2, (255,0,0), -1)

cv2.imshow("corners", base2)
cv2.imshow("base", base)
cv2.waitKey(0)
cv2.destroyAllWindows()