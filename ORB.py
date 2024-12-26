import cv2

img = cv2.imread("lena.jpg")

orb = cv2.ORB_create()

orb.setMaxFeatures(50)

kp, desc = orb.detectAndCompute(img, None)

cv2.drawKeypoints(img, kp, img, (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()