import cv2

img = cv2.imread("lena.jpg")

agast = cv2.AgastFeatureDetector_create()

kp = agast.detect(img, None)

cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()