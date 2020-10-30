import cv2
import pandas as pd
import os

def add_photos_dir(source_dir, has_item):
    if source_dir[-1] != '/':
        source_dir += '/'
    if has_item:
        source_dir += 'item_pics'
    else:
        source_dir += 'non_item_pics'
    return  source_dir


def read_imgs(source_dir, has_item, return_names=False):
    source_dir = add_photos_dir(source_dir, has_item)
    images = []
    img_names = os.listdir(source_dir)
    for img_name in img_names:
        img = cv2.imread(f'{source_dir}/{img_name}')
        images.append(img)
    if return_names:
        return images, img_names
    else:
        return images

def read_results(file_name):
    abs_path = os.path.dirname(os.path.realpath(__file__))
    if not abs_path.endswith('/'):
        abs_path += '/'
    return pd.read_pickle(f'{abs_path}../results/{file_name}')