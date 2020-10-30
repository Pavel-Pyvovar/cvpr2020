import cv2
import numpy as np
import os
from tqdm import tqdm
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
if not os.path.isdir(f'{ROOT_DIR}/preprocessed_data'):
    os.mkdir(f'{ROOT_DIR}/preprocessed_data')

if not os.path.isdir(f'{ROOT_DIR}/preprocessed_data/{DATA_SET}'):
    os.mkdir(f'{ROOT_DIR}/preprocessed_data/{DATA_SET}')

if not os.path.isdir(f'{ROOT_DIR}/preprocessed_data/{DATA_SET}/{photos_dirname}'):
    os.mkdir(f'{ROOT_DIR}/preprocessed_data/{DATA_SET}/{photos_dirname}')

if not os.path.isdir(f'{ROOT_DIR}/preprocessed_data/{DATA_SET}/{photos_dirname}/item_pics'):
    os.mkdir(f'{ROOT_DIR}/preprocessed_data/{DATA_SET}/{photos_dirname}/item_pics')
    os.mkdir(f'{ROOT_DIR}/preprocessed_data/{DATA_SET}/{photos_dirname}/non_item_pics')


def cut_into_grid(img, patch_height, patch_width):
    """Function to cut image into a grid of patches.

    Args:
        img: numpy.array, input image
        patch_height: height of the patch
        patch_width: width of the patch

    Returns:
        patches: list of patches
    """
    patches = []
    n_vert_patches = img.shape[0] // patch_height # 480 / 16 = 30
    n_horiz_patches = img.shape[1] // patch_width # 640 / 16 = 40

    for i in range(n_vert_patches):
        for j in range(n_horiz_patches):
            patch_ij = img[i*patch_height:i*patch_height + patch_height,
                            j*patch_height:j*patch_height + patch_width]
            patches.append(patch_ij)
    return patches


def preprocess_imgs(source_dir, dest_dir, patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, has_item=HAS_ITEM):
    imgs = read_imgs(source_dir, has_item)
    dest_dir = add_photos_dir(dest_dir, has_item)
    for i, img in tqdm(enumerate(imgs[:5])):
        patches = cut_into_grid(img, patch_height, patch_width)
        for j, patch in enumerate(patches):
            patch_name = f'image{i}_patch_{j}'
            cv2.imwrite(f'{dest_dir}/{patch_name}.jpg', patch)


source_dir = f'{ROOT_DIR}/data/{DATA_SET}/resized/photos_{ORIGINAL_WIDTH}/'
dest_dir = f'{ROOT_DIR}/preprocessed_data/{DATA_SET}/{photos_dirname}/'

preprocess_imgs(source_dir, dest_dir)
# imgs = read_imgs(source_dir)
