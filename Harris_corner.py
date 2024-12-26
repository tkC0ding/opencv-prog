import cv2
import numpy as np

base = np.zeros((480, 640, 3), dtype=np.uint8)
base2 = base.copy()

cv2.rectangle(base, (100, 100), (200, 200), (255, 255, 255), -1)

gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

corners = cv2.cornerHarris(gray, 2, 3, 0.04)

base2[corners>0.01*corners.max()]=[255,255,255]

cv2.imshow("corners", base2)
cv2.imshow("base", base)
cv2.waitKey(0)
cv2.destroyAllWindows()