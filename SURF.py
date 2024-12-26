import cv2

img = cv2.imread("lena.jpg")

surf = cv2.xfeatures2d.SURF_create(3000)

kp = surf.detect(img, None)

cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()