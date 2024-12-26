import cv2
import numpy as np


base = np.zeros((480, 640, 3)).astype(np.uint8)
base2 = base.copy()

cv2.rectangle(base, (100,100), (200,200), (255,0,0), -1)

canny_img = cv2.Canny(base, 50, 150)

hough_lines = cv2.HoughLines(canny_img, 1, np.pi/180, 80)

for line in hough_lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = rho*a
        y0 = rho*b

        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        cv2.line(base2, pt1, pt2, (255,255,255), 2, cv2.LINE_AA)

cv2.imshow("Hough", base2)
cv2.imshow("edge", canny_img)
cv2.imshow("image", base)
cv2.waitKey(0)
cv2.destroyAllWindows()