import cv2
import numpy as np

base = np.zeros((480, 640, 3), dtype=np.uint8)
base2 = base.copy()

cv2.rectangle(base, (100, 100), (200, 200), (255, 0, 0), -1)
cv2.circle(base, (50,50), 50, (255, 0, 0), -1)

canny_img = cv2.Canny(base, 50, 150)

hough_lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, 50)

for lines in hough_lines:
    x1, y1, x2, y2 = lines[0]
    cv2.line(base2, (x1,y1), (x2,y2), (255,255,255), 2)

cv2.imshow("hough", base2)
cv2.imshow("edges", canny_img)
cv2.imshow("base", base)
cv2.waitKey(0)
cv2.destroyAllWindows()