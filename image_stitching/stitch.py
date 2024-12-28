import cv2

images = []
for i in range(1, 4):
    a =  cv2.imread(f"img_{i}.jpg")
    a = cv2.resize(a, (0,0),fx=0.8, fy=0.6)
    images.append(a)

stitcher = cv2.Stitcher.create()

check, out = stitcher.stitch(images)

for i,j in enumerate(images):
    cv2.imshow(f"img_{i}", j)
cv2.imshow("panorama", out)
cv2.waitKey(0)
cv2.destroyAllWindows()