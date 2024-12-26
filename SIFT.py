import cv2

image = cv2.imread("lena.jpg")

sift = cv2.SIFT_create()
sift.setNFeatures(500)
sift.setContrastThreshold(0.03)
sift.setEdgeThreshold(10)

keypoints = sift.detect(image, None)

cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()