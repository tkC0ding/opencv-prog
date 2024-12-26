import cv2
import numpy as np

img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(img, (5,5), 10)

kernel = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]
)

lap = cv2.Laplacian(blur, -1) * (255/56)
filter_img = cv2.filter2D(blur, -1, kernel=kernel) * (255/56)

cv2.imshow("LoG", lap)
cv2.imshow("img", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()