import numpy as np
import cv2


img = cv2.imread("lena.jpg")

gaussian_1 = cv2.GaussianBlur(img, (5,5), 2)

gaussian_2 = cv2.GaussianBlur(img, (5,5), 2.828)

D0G = gaussian_2 - gaussian_1


cv2.imshow("DoG", D0G)
cv2.imshow("image_1", gaussian_1)
cv2.imshow("image_2", gaussian_2)
cv2.waitKey(0)
cv2.destroyAllWindows()