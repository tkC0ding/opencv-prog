import cv2

temp = cv2.imread("temp.jpg")
test = cv2.imread("test.jpg")


temp = cv2.resize(temp, ((temp.shape[1]//2) - 1, (temp.shape[0]//2)))
test = cv2.resize(test, ((test.shape[1]//2), (test.shape[0]//2) - 1))

matcher = cv2.BFMatcher()
orb = cv2.ORB_create()

kp_temp, desc_temp = orb.detectAndCompute(temp, None)
kp_test, desc_test = orb.detectAndCompute(test, None)

matches = matcher.knnMatch(desc_temp, desc_test, k=2)

matches_ = []
matched_keypoints = []
for m,n in matches:
    if(m.distance < 0.8*n.distance):
        matches_.append(m)
        matched_keypoints.append(kp_test[m.trainIdx])

img3 = cv2.drawMatches(temp, kp_temp, test, kp_test, matches_, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

cv2.drawKeypoints(test, matched_keypoints, test, (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("test", test)
cv2.imshow("matches", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()