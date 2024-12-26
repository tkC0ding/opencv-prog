import cv2

img = cv2.imread("lena.jpg")

fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()


fast.setThreshold(100)
fast.setNonmaxSuppression(True)
kp = fast.detect(img, None)

kp, desc = brief.compute(img, kp)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()