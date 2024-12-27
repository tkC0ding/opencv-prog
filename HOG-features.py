import cv2
from skimage.feature import hog


img = cv2.imread("lena.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = img[100:400, 200:350]

img = cv2.resize(img, (64, 128))

hog_features, hog_image = hog(img, 
                              orientations=9, 
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), 
                              block_norm='L2-Hys', #list of norms : L2_Norm, L2_Hys, L1_Norm, L1_sqrt
                              visualize=True, 
                              transform_sqrt=True)

cv2.imshow("hog", hog_image)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()