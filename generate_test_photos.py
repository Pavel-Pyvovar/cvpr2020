from os import error
import cv2 as cv
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt 

descriptors_to_test = {'ORB(nfeatures=500, scoreType=ORB_HARRIS_SCORE)': cv.ORB_create(nfeatures=500, scoreType=cv.ORB_HARRIS_SCORE),
                       'ORB(nfeatures=500, scoreType=ORB_FAST_SCORE)': cv.ORB_create(nfeatures=500, scoreType=cv.ORB_FAST_SCORE),
                       'ORB(nfeatures=250, scoreType=ORB_HARRIS_SCORE)': cv.ORB_create(nfeatures=250),
                       'BRISK(thresh=10)': cv.BRISK_create(10), 'BRISK(thresh=20)': cv.BRISK_create(20), 'BRISK(thresh=25)': cv.BRISK_create(25),
                       'BRISK(thresh=30)': cv.BRISK_create(30), 'BRISK(thresh=40)': cv.BRISK_create(40),
                       'BRIEF()': cv.xfeatures2d.BriefDescriptorExtractor_create()}

train_images = {'ppe':f'photos_resized_pavlo/photos_{2048}/item_pics/20200918_161314.jpg',
                'zvv':f'photos_resized_zvv/item_pics/photos_{2048}/P00930-164836_1.jpg', 
                'vvr':f'photos_resized_vvr/item_pics/photos_{2048}/IMG_20201003_153648.jpg'}

test_images_item = {'ppe':f'photos_resized_pavlo/photos_{2048}/item_pics/20200918_153021.jpg',
                'zvv':f'photos_resized_zvv/item_pics/photos_{2048}/P00930-164730.jpg', 
                'vvr':f'photos_resized_vvr/item_pics/photos_{2048}/IMG_20201003_153826_1.jpg'}

test_images_no_item = {'ppe':f'photos_resized_pavlo/photos_{2048}/non_item_pics/20200918_192013.jpg',
                       'zvv':f'photos_resized_zvv/no_item_pics/photos_{2048}/P00930-171047.jpg', 
                       'vvr':f'photos_resized_vvr/no_item_pics/photos_{2048}/IMG_20201003_154658.jpg'}




def save_img_resule(matches, img1, img2, kp1, kp2, dataset, descriptor, items):
    matches = sorted(matches, key = lambda x:x.distance)
    try:
        img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv.imwrite(f'./result_pics/{dataset}-{descriptor}-{items}.png', img3)

    except Exception as e:
        print(descriptor, dataset, e)

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
        return test_kp, [], test_img
    return test_kp, matches, test_img


def process_images(train_img_path, test_img_path, dataset, items):

    train_img = cv.imread(train_img_path, cv.IMREAD_GRAYSCALE)

    for descriptor_name in tqdm(descriptors_to_test.keys()):
        descriptor = descriptors_to_test[descriptor_name]

        if not isinstance(descriptor, cv.xfeatures2d_BriefDescriptorExtractor):
            train_kp, train_descriptors = descriptor.detectAndCompute(
                train_img, None)
        else:
            # needed for BRIEF descriptor
            star = cv.xfeatures2d.StarDetector_create()
            star_kp = star.detect(train_img, None)
            train_kp, train_descriptors = descriptor.compute(
                train_img, star_kp)

        test_kp, matches, test_img = test_image(train_descriptors, descriptor, test_img_path)

        save_img_resule(matches, train_img, test_img, train_kp, test_kp, dataset, descriptor_name.lower(), items)

for dataset in ['zvv', 'vvr']:
    process_images(train_images[dataset], test_images_item[dataset], dataset, 'item')

for dataset in ['zvv', 'vvr']:
    try:
        process_images(train_images[dataset], test_images_no_item[dataset], dataset, 'NOitem')
    except Exception as e:
        print(e, dataset)
