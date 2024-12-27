import cv2

temp = cv2.imread("temp.jpg")
test_img = cv2.imread("test.jpg")

temp = cv2.resize(temp, ((temp.shape[1]//2) - 1, (temp.shape[0]//2)))
test_img = cv2.resize(test_img, ((test_img.shape[1]//2), (test_img.shape[0]//2) - 1))

orb = cv2.ORB_create()
k = cv2.KeyPoint()

kp_temp, desc_temp = orb.detectAndCompute(temp, None)
kp_test, desc_test = orb.detectAndCompute(test_img, None)

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = matcher.match(desc_temp, desc_test, None)

img3 = cv2.drawMatches(temp, kp_temp, test_img, kp_test, matches, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

matched_keypoints = []
for match in matches:
    matched_keypoints.append(kp_test[match.trainIdx])

matched_keypoints = tuple(matched_keypoints)

cv2.drawKeypoints(test_img, matched_keypoints, test_img, (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("matches", img3)
cv2.imshow("test", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()