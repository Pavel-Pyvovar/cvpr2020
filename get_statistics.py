# from matplotlib import pyplot as plt
import time

import cv2 as cv
import pandas as pd
from tqdm import tqdm
from pathlib import Path
descriptors_to_test = {'ORB(nfeatures=500, scoreType=ORB_HARRIS_SCORE)': cv.ORB_create(nfeatures=500, scoreType=cv.ORB_HARRIS_SCORE),
                       'ORB(nfeatures=500, scoreType=ORB_FAST_SCORE)': cv.ORB_create(nfeatures=500, scoreType=cv.ORB_FAST_SCORE),
                       'ORB(nfeatures=250, scoreType=ORB_HARRIS_SCORE)': cv.ORB_create(nfeatures=250),
                       'BRISK(thresh=10)': cv.BRISK_create(10), 'BRISK(thresh=20)': cv.BRISK_create(20), 'BRISK(thresh=25)': cv.BRISK_create(25),
                       'BRISK(thresh=30)': cv.BRISK_create(30), 'BRISK(thresh=40)': cv.BRISK_create(40),
                       'BRIEF()': cv.xfeatures2d.BriefDescriptorExtractor_create()}


def test_image(trained_descriptors,  descriptor, test_img_path):
    """Processes test image given a descriptor and descriptor's train features
       and counts some statistics
    Args:
        trained_descriptors (np.array): descriptor matrix of a train image
        descriptor (cv2.ORB or cv2.BRISK or cv2.xfeatures2d_BriefDescriptorExtractor): descriptor to test
        test_img_path (str): test image path

    Returns:
        (float, float): mateched_kp_proportion and avg_kp_difference respectively
    """
    test_img = cv.imread(str(test_img_path), cv.IMREAD_GRAYSCALE)
    if not isinstance(descriptor, cv.xfeatures2d_BriefDescriptorExtractor):
        test_kp, test_descriptors = descriptor.detectAndCompute(test_img, None)
    else:
        # needed for BRIEF descriptor
        star = cv.xfeatures2d.StarDetector_create()
        star_kp = star.detect(test_img, None)
        test_kp, test_descriptors = descriptor.compute(test_img, star_kp)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(trained_descriptors, test_descriptors)
    except Exception as e:
        # print(e, test_descriptors)
        return 0, -1

    mateched_kp = len(matches)/len(trained_descriptors)
    avg_kp_difference = sum(x.distance for x in matches)/len(matches)
    return mateched_kp, avg_kp_difference


def process_images(train_img_path, paths):
    descriptor_results = {descr: {} for descr in descriptors_to_test.keys()}

    train_img = cv.imread(train_img_path, cv.IMREAD_GRAYSCALE)

    for descriptor_name in tqdm(descriptors_to_test.keys()):
        descriptor = descriptors_to_test[descriptor_name]

        start_time = time.time()
        if not isinstance(descriptor, cv.xfeatures2d_BriefDescriptorExtractor):
            train_kp, train_descriptors = descriptor.detectAndCompute(
                train_img, None)
        else:
            # needed for BRIEF descriptor
            star = cv.xfeatures2d.StarDetector_create()
            star_kp = star.detect(train_img, None)
            train_kp, train_descriptors = descriptor.compute(
                train_img, star_kp)

        mateched_kps = []
        avg_kp_differences = []
        for img_path in paths:
            res = test_image(train_descriptors, descriptor, img_path)
            mateched_kps.append(res[0])
            avg_kp_differences.append(res[1])

        descriptor_results[descriptor_name]['mateched_kps'] = sum(
            mateched_kps)/len(mateched_kps)
        avg_kp_differences = [x for x in avg_kp_differences if x!=-1]
        descriptor_results[descriptor_name]['avg_kp_differences'] = sum(
            avg_kp_differences)/len(avg_kp_differences)
        descriptor_results[descriptor_name]['avg_time'] = (
            time.time() - start_time)/len(paths)
    return descriptor_results

train = '/home/vzalevskyi/git/cvpr2020/photos_resized_pavlo/photos_640/20200918_101345.jpg'
item_pics = list(Path().cwd().joinpath('photos_resized_pavlo/photos_640/').iterdir())

res = process_images(train, item_pics)
pd.DataFrame(res).T
