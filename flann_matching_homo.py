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
# The number of bits to shift to check for neighboring buckets
# (0 is regular LSH, 2 is recommended).
MULTI_PROBE_LEVEL = 1 # 2
MIN_MATCH_COUNT = 10

cv2.namedWindow("img", cv2.WINDOW_NORMAL)

images = os.listdir('photos')
image1 = cv2.imread(f'photos/{images[30]}')
image2 = cv2.imread(f'photos/{images[31]}')

orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

index_params= {"algorithm": FLANN_INDEX_LSH,
                "table_number": TABLE_NUMBER,
                "key_size": KEY_SIZE,
                "multi_probe_level": MULTI_PROBE_LEVEL}

# checks: The number of times the tree(s) in the index should be recursively
# traversed. A higher value for this parameter would give better search 
# precision, but also take more time. If automatic configuration was used
# when the index was created, the number of checks required to achieve 
# the specified precision was also computed, in which case this parameter 
# is ignored.

search_params = {"checks": 100}

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
if len(good_matches) >= MIN_MATCH_COUNT:
    # Coordinates of the points in the original plane.
    source_points = [keypoints1[m.queryIdx].pt for m in good_matches]
    source_points = np.float32(source_points).reshape(-1, 1, 2)
    # Coordinates of the points in the target plane.
    destination_points = [keypoints2[m.queryIdx].pt for m in good_matches]
    destination_points = np.float32(destination_points).reshape(-1, 1, 2)

    # Find a perspective transformation between two planes.
    # Docs: https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga4b3841447530523e5272ec05c5d1e411
    homography_params = {
        # Method used to compute a homography matrix
        "method": cv2.RANSAC,
        # Maximum allowed reprojection error to treat a point pair as an inlier
        "ransacReprojThreshold": 5.0
    }
    retval, mask = cv2.findHomography(source_points, destination_points,
                                    **homography_params)
    matches_mask = list(mask.ravel())
    h, w, d = image1.shape
    points = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
   
    # Perform the perspective matrix transformation of vectors.
    # Docs: https://docs.opencv.org/master/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
    destination = cv2.perspectiveTransform(points, retval)
    
    # Draw several polygonal curves.
    polylines_draw_params = {
        "color": (0, 255, 0),
        "thickness": 255,
        # Type of the line segments
        "lineType": 3,
        # Number of fractional bits in the vertex coordinates.
        "shift": cv2.LINE_AA
    }
    image2 = cv2.polylines(image2,
                # Array of polygonal curves
                [np.int32(destination)], 
                # Flag indicating whether the drawn polylines are closed
                # or not. If they are closed, the function draws a line 
                # from the last vertex of each curve to its first vertex.
                True, **polylines_draw_params)
else:
    print( "Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT) )
    matches_mask = None

img_matches = np.empty((max(image1.shape[0], image2.shape[0]),
                        image1.shape[1]+image2.shape[1], 3), dtype=np.uint8)

draw_matches_params = {
    "matchColor": (0, 255, 0),
    "singlePointColor": None,
    "matchesMask": matches_mask,
    "flags": cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS # 2
}
img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches,
                    outImg=None, **draw_matches_params)
# cv2.drawKeypoints(image1, keypoints, image1, color=(0, 255, 0), flags=0)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()