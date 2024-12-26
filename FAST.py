import cv2

img = cv2.imread("lena.jpg")
img_2 = img.copy()

fast = cv2.FastFeatureDetector_create()

fast.setThreshold(100)
fast.setNonmaxSuppression(True)
kp = fast.detect(img)
cv2.drawKeypoints(img, kp, img, (255,255,255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)

fast.setNonmaxSuppression(False)
kp = fast.detect(img_2)
cv2.drawKeypoints(img_2, kp, img_2, (255,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

cv2.imshow("lena_2", img_2)
cv2.imshow("lena", img)
cv2.waitKey(0)
cv2.destroyAllWindows()