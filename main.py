import cv2
import os
import numpy as np

NTREES = 5
FLANN_INDEX_KDTREE = 0
# cv2.namedWindow("img", cv2.WINDOW_NORMAL)


# img1 = cv2.imread(f"photos/{os.listdir('photos')[2]}", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(f"photos/{os.listdir('photos')[4]}", cv2.IMREAD_GRAYSCALE)

# orb = cv2.ORB_create()
# orb.empty()

# keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
# keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# # img2 = cv2.drawKeypoints(img, keypoints, np.array([]), color=(255, 0, 0), flags=0)

# bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING)
# # matches = bfmatcher.match(descriptors1, descriptors2)
# matches = bfmatcher.knnMatch(descriptors1, descriptors2, k=2)

# # matches = sorted(matches, key=lambda x: x.distance)

# good = []
# for m, n in matches:
#     if m.distance < .75 * n.distance:
#         good.append([m])

# # img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, np.array([]), flags=2)
# img = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, np.array([]), flags=2)

# # FLANN parameters
# # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = NTREES)
# # search_params = dict(checks=100)   # or pass empty dictionary

# # flann = cv2.FlannBasedMatcher(index_params, search_params)

# # matches = flann.knnMatch(np.asarray(descriptors1,np.float32), 
# #                                     np.asarray(descriptors2, np.float32), k=2)

# # # Need to draw only good matches, so create a mask
# # matchesMask = [[0,0] for i in range(len(matches))]

# # # ratio test as per Lowe's paper
# # for i,(m,n) in enumerate(matches):
# #     if m.distance < 0.7*n.distance:
# #         matchesMask[i]=[1,0]

# # draw_params = dict(matchColor = (0,255,0),
# #                    singlePointColor = (255,0,0),
# #                    matchesMask = matchesMask,
# #                    flags = 0)

# # img = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2,
# #                              matches, None, **draw_params)



# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def gen_descriptors():
    """Function for generating object descriptors.
    """
    keypoints, descriptors = [], []
    for img_name in os.listdir('photos'):
        image = cv2.imread(f"photos/{img_name}", cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()
        orb.empty()
        kp, des = orb.detectAndCompute(image, None)
        keypoints.append(kp)
        descriptors.append(des)
    return keypoints, descriptors


if __name__ == "__main__":
    keypoints, descriptors = gen_descriptors()    
    print(np.array(keypoints).shape, np.array(descriptors).shape)


