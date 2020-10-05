import cv2
import os
import numpy as np
from pprint import pprint

cv2.namedWindow("img", cv2.WINDOW_NORMAL)

images = os.listdir('photos')
image1 = cv2.imread(f'photos/{images[4]}')
image2 = cv2.imread(f'photos/{images[5]}')

orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
good = []
for n, m in matches:
    if n.distance < .75 * m.distance:
        good.append([n])


# pprint(matches)
# print([match.distance for match in matches])
# print(len(matches))

img = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good, 
                        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.drawKeypoints(image1, keypoints, image1, color=(0, 255, 0), flags=0)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()