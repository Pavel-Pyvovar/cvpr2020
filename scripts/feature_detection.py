import cv2
import os
import pandas as pd
from utilities import read_imgs, add_photos_dir

# Constants
ROOT_DIR = f'{os.path.dirname(os.path.realpath(__file__))}/..'
DATA_SET = 'ppe'
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640
PATCH_HEIGHT = 60
PATCH_WIDTH = 80
HAS_ITEM = True

photos_dirname = f'{ORIGINAL_HEIGHT}_{ORIGINAL_WIDTH}_to_{PATCH_HEIGHT}_{PATCH_WIDTH}'
source_dir = f'{ROOT_DIR}/preprocessed_data/{DATA_SET}/{photos_dirname}/'

descriptors = dict(ORB = cv2.ORB_create(),
                   BRISK = cv2.BRISK_create(),
                    SIFT = cv2.SIFT_create())
                   # BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create())

def detect_features(imgs, img_names, descriptor, has_item):
    """Feature detection and represenation.

    Args:
        img: input image
        descriptor: algorithm used to detect features.

    Returns:

    """
    results = []
    descriptor_name = descriptor.getDefaultName().split('.')[-1]

    if has_item:
        target = f'{DATA_SET}_item'
    else:
        target = f'{DATA_SET}_non_item'

    for i, img in enumerate(imgs):
        keypoints, descriptions = descriptor.detectAndCompute(img, None)
        if len(keypoints) == 0:
            break
        else:
            results.append(dict(img_names = img_names[i],
                                 decriptor_values = descriptions,
                                 target = target,
                                 descriptor_names = descriptor_name))
    return results


def run_descriptors(descriptors, source_dir, has_item=HAS_ITEM):
    total_results = []
    imgs, names = read_imgs(source_dir, has_item, return_names=True)
    for descriptor_name in descriptors:
        descriptor = descriptors[descriptor_name]
        if descriptor_name == 'BRIEF':
            # There is a problem with xfeatures2d so I decided to skip this part.
            # There is a potential bug here.
            star = cv2.xfeatures2d.StarDetector_create()
            results = detect_features(imgs, names, descriptor, has_item)
        else:
            results = detect_features(imgs, names, descriptor, has_item)
        total_results += results
    pd.DataFrame(total_results).to_pickle(f'{ROOT_DIR}/results/descriptions.pkl')


orb = cv2.ORB_create()
brisk = cv2.BRISK_create()
sift = cv2.SIFT_create()

imgs, names = read_imgs(source_dir, HAS_ITEM, return_names=True)
# print(len(imgs))
# results = detect_features(imgs, names, brisk, HAS_ITEM)
# print(results)
run_descriptors(descriptors, source_dir)
# print(results)