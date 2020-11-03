# from matplotlib import pyplot as plt
import time

import cv2 as cv
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
descriptors_to_test = {'ORB(nfeatures=500, scoreType=ORB_HARRIS_SCORE)': cv.ORB_create(nfeatures=500, scoreType=cv.ORB_HARRIS_SCORE),
                       'ORB(nfeatures=500, scoreType=ORB_FAST_SCORE)': cv.ORB_create(nfeatures=500, scoreType=cv.ORB_FAST_SCORE),
                       'ORB(nfeatures=250, scoreType=ORB_HARRIS_SCORE)': cv.ORB_create(nfeatures=250),
                       'BRISK(thresh=10)': cv.BRISK_create(10), 'BRISK(thresh=20)': cv.BRISK_create(20), 'BRISK(thresh=25)': cv.BRISK_create(25),
                       'BRISK(thresh=30)': cv.BRISK_create(30), 'BRISK(thresh=40)': cv.BRISK_create(40),
                       'BRIEF()': cv.xfeatures2d.BriefDescriptorExtractor_create()}


def test_image(trained_descriptors,  descriptor, test_img_path, from_frame=False):
    """Processes test image given a descriptor and descriptor's train features
       and counts some statistics
    Args:
        trained_descriptors (np.array): descriptor matrix of a train image
        descriptor (cv2.ORB or cv2.BRISK or cv2.xfeatures2d_BriefDescriptorExtractor): descriptor to test
        test_img_path (str): test image path

    Returns:
        (float, float): mateched_kp_proportion and avg_kp_difference respectively
    """
    if from_frame:
        test_img = test_img_path
    else:
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
        #choose top 100 mathced key points and get their idxs
        top100_idxs = [x.trainIdx for x in sorted(matches, key = lambda x:x.distance)[:100]]
    
    except Exception as e:
        if test_descriptors:
            top100_idxs = list(range(1,min(len(test_descriptors), 101)))
        else:
            return np.array([])

    return np.array(test_descriptors)[top100_idxs]


def process_images(train_img_path, paths, target):
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

        top_descriptors = []
        img_names = []
        for img_path in paths:
            res = test_image(train_descriptors, descriptor, img_path)
            top_descriptors.append(res)
            img_names.append(img_path)
        
        descriptor_results[descriptor_name]['top_descriptors'] = top_descriptors
        descriptor_results[descriptor_name]['img_names'] = img_names

    result_df = pd.DataFrame(descriptor_results).T
    result_df['target'] = target
    return result_df

size = 2048


# if __name__ == '__main__':
#     name = 'ppe'
#     train = f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/photos_{size}/item_pics/20200918_161314.jpg'
#     item_pics = list(Path().cwd().joinpath(f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/photos_{size}/item_pics').iterdir())
#     ppe_items = process_images(train, item_pics, 'ppe_item')

#     train = f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/photos_{size}/item_pics/20200918_161314.jpg'
#     item_pics = list(Path().cwd().joinpath(f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/photos_{size}/non_item_pics').iterdir())
#     ppe_no_items = process_images(train, item_pics, 'ppe_non_item')






#     name = 'zvv'
#     train = f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/item_pics/photos_{size}/P00930-164836_1.jpg'
#     item_pics = list(Path().cwd().joinpath(f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/item_pics/photos_{size}').iterdir())
#     zvv_items = process_images(train, item_pics, 'zvv_item')

#     train = f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/item_pics/photos_{size}/P00930-164836_1.jpg'
#     item_pics = list(Path().cwd().joinpath(f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/non_item_pics/photos_{size}').iterdir())
#     zvv_no_items = process_images(train, item_pics, 'zvv_non_item')






#     name = 'rvv'
#     train = f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/item_pics/photos_{size}/IMG_20201003_153648.jpg'
#     item_pics = list(Path().cwd().joinpath(f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/item_pics/photos_{size}').iterdir())
#     rvv_items = process_images(train, item_pics, 'rvv_item')

#     train = f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/item_pics/photos_{size}/IMG_20201003_153648.jpg'
#     item_pics = list(Path().cwd().joinpath(f'/home/vzalevskyi/git/cvpr2020/data/{name}/resized/non_item_pics/photos_{size}').iterdir())
#     rvv_no_items = process_images(train, item_pics, 'rvv_non_item')






#     final_result_df = {'img_names':[], 'descriptor_values':[], 'target':[],
#                     'descriptor_names':[]}

#     datasets = [ppe_items, ppe_no_items, zvv_items, zvv_no_items, rvv_items, rvv_no_items]
#     for dataset in datasets:
#         for descr_name in dataset.index.values:
#             for idx, descr_val in enumerate(dataset.loc[descr_name]['top_descriptors']):
#                 final_result_df['descriptor_values'].append(descr_val)
#                 final_result_df['img_names'].append(str(dataset.loc[descr_name]['img_names'][idx]).split('/')[-1])
#                 final_result_df['target'].append(dataset.loc[descr_name]['target'])
#                 final_result_df['descriptor_names'].append(descr_name)
                
#     final_result_df = pd.DataFrame(final_result_df)
#     final_result_df.to_pickle('../results/image_descriptions.pkl')

# import pandas as pd
# final_result_df = pd.read_pickle('../results/image_descriptions.pkl')
