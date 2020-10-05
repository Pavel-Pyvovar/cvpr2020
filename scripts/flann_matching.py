import cv2
import os
import numpy as np
from pprint import pprint

# The threshold for ratio test value described in Lowe at al. (2004).
RATIO_THRESHOLD = .75
FLANN_INDEX_LSH = 6
# The number of hash tables to use (between 10 and 30 usually).
TABLE_NUMBER = 6 # 12
# The size of the hash key in bits (between 10 and 20 usually).
KEY_SIZE = 12 # 20
# The number of bits to shift to check for neighboring buckets (0 is regular LSH, 2 is recommended).
MULTI_PROBE_LEVEL = 1 # 2

cv2.namedWindow("img", cv2.WINDOW_NORMAL)

images = os.listdir('photos')
image1 = cv2.imread(f'photos/{images[0]}')
image2 = cv2.imread(f'photos/{images[10]}')

orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = TABLE_NUMBER,
                   key_size = KEY_SIZE,
                   multi_probe_level = MULTI_PROBE_LEVEL)

# checks: The number of times the tree(s) in the index should be recursively
# traversed. A higher value for this parameter would give better search 
# precision, but also take more time. If automatic configuration was used
# when the index was created, the number of checks required to achieve 
# the specified precision was also computed, in which case this parameter 
# is ignored.

search_params = dict(checks=100)

flann = cv2.FlannBasedMatcher(index_params, search_params)
# flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

# Note that in the following line of code descriptors are converted into float32!
# This is neccessary because the following issue arises if typecast is not done:
# cv2.error: OpenCV(4.4.0) /tmp/pip-req-build-6179nsls/opencv/modules/flann/src/miniflann.cpp:315: error: (-210:Unsupported format or combination of formats) in function 'buildIndex_'
# In other words: Flann needs the descriptors to be of type CV_32F 
# This issue is addressed on the OpenCV forum: https://answers.opencv.org/question/11209/unsupported-format-or-combination-of-formats-in-buildindex-using-flann-algorithm/?sort=latest

matches = flann.knnMatch(descriptors1, descriptors2, k=2)
matches = [match for match in matches if len(match) == 2]
print(len(matches))

good_matches = []
for m, n in matches:
    if m.distance < RATIO_THRESHOLD * n.distance:
        good_matches.append(m)

print(len(good_matches))
# img_matches = np.empty((max(image1.shape[0], image2.shape[0]),
#                         image1.shape[1]+image2.shape[1], 3), dtype=np.uint8)
img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches,
                    None, matchColor=(0, 255, 0), flags=2)
# cv2.drawKeypoints(image1, keypoints, image1, color=(0, 255, 0), flags=0)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()