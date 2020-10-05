import cv2
import os
import numpy as np
import csv
import pickle
from time import time
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

class ORBEvaluator:
    'Class for generating descriptors, and collecting metrics'

    def __init__(self, ratio_threshold=.75, flann_index_lsh = 6,
                table_number=6, key_size=12, multi_probe_level=1):
        self.ratio_threshold = ratio_threshold
        self.flann_index_lsh = flann_index_lsh
        self.table_number = table_number
        self.key_size = key_size
        self.multi_probe_level = multi_probe_level
        

    def load_images(self, dirname):
        img_names = os.listdir(f'{dirname}')[:10]
        name_img_map = {}
        for img_name in img_names:
            name_img_map[img_name] = cv2.imread(f'{dirname}/{img_name}') 
        return name_img_map


    def get_descriptors(self, name_img_mapping):
        imgname_desctime_map = {}
        imgname_descriptors_map = {}
        imgname_keypoints_map = {}
        orb = cv2.ORB_create()
        for img_name in name_img_mapping:
            img = name_img_mapping[img_name]
            start = time()
            keypoints, descriptors = orb.detectAndCompute(img, None)
            end = time()            
            imgname_desctime_map[img_name] = end - start
            imgname_descriptors_map[img_name] = descriptors
            imgname_keypoints_map[img_name] = keypoints
        return imgname_desctime_map, imgname_descriptors_map, imgname_keypoints_map


    def get_matching_metrics(self, imgname_descriptors_map, imgname_keypoints_map):
        matching_metrics = []
        for img1_name in imgname_descriptors_map:
            for img2_name in imgname_descriptors_map:
                if img1_name != img2_name:
                    descriptors1 = imgname_descriptors_map[img1_name]
                    descriptors2 = imgname_descriptors_map[img2_name]
                    keypoints1 = imgname_keypoints_map[img1_name]
                    keypoints2 = imgname_keypoints_map[img2_name]
                    start = time()
                    good_matches, good_matches_percent = self.matcher(descriptors1, descriptors1)
                    end = time()
                    #############################################
                    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                    # img1 = cv2.imread(f"photos/{img1_name}")
                    # img2 = cv2.imread(f"photos/{img2_name}")
                    # img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=2)
                    # cv2.imshow('img', img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    #############################################
                    matching_metrics.append({
                        "first_image": img1_name,
                        "second_image": img2_name,
                        "average_distance": np.mean([match.distance for match in good_matches]),
                        "matches": len(good_matches),
                        "good_matches_percent": good_matches_percent,
                        "matching_time": end - start
                    })
        return matching_metrics


    def matcher(self, descriptors1, descriptors2):
        index_params= {"algorithm": self.flann_index_lsh,
                        "table_number": self.table_number,
                        "key_size": self.key_size,
                        "multi_probe_level": self.multi_probe_level}

        # checks: The number of times the tree(s) in the index should be recursively
        # traversed. A higher value for this parameter would give better search 
        # precision, but also take more time. If automatic configuration was used
        # when the index was created, the number of checks required to achieve 
        # the specified precision was also computed, in which case this parameter 
        # is ignored.

        search_params = {"checks": 100}

        # FLANN (Fast Library for Approximate Nearest Neighbors) is a 
        # library that contains a collection of algorithms optimized for
        # fast nearest neighbor search in large datasets and for high 
        # dimensional features. More information about FLANN can be found
        # in Marius Muja, David G. Lowe. Fast Approximate Nearest Neighbors
        # with Automatic Algorithm Configuration, 2009.

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

        # Note that in the following line of code descriptors are converted into float32!
        # This is neccessary because the following issue arises if typecast is not done:
        # cv2.error: OpenCV(4.4.0) /tmp/pip-req-build-6179nsls/opencv/modules/flann/src/miniflann.cpp:315: error: (-210:Unsupported format or combination of formats) in function 'buildIndex_'
        # In other words: Flann needs the descriptors to be of type CV_32F 
        # This issue is addressed on the OpenCV forum: https://answers.opencv.org/question/11209/unsupported-format-or-combination-of-formats-in-buildindex-using-flann-algorithm/?sort=latest

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        matches = [match for match in matches if len(match) == 2]

        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
        print(len(good_matches), len(matches))
        good_matches_percent = len(good_matches) / len(matches) * 100
        return good_matches, good_matches_percent


    def write_metrics(self, dirname):
        descriptors_time, img_descriptors = self.get_descriptors(dirname)
        distances = self.get_distances(img_descriptors)
        metrics = {
            "descriptors_time": descriptors_time,
            "distances": distances
        }

        with open('metrics.pkl', 'wb') as file:
            pickle.dump(metrics, file)


evaluator = ORBEvaluator()
photos = evaluator.load_images('photos')
imgname_desctime_map, imgname_descriptors_map, imgname_keypoints_map = evaluator.get_descriptors(photos)
matching_metrics = evaluator.get_matching_metrics(imgname_descriptors_map, imgname_keypoints_map)
pprint(matching_metrics)


# processor.write_metrics('photos')

# with open('metrics.pkl', 'rb') as file:
#     metrics = pickle.load(file)

# pprint(metrics)
