import os
import multiprocessing as mp
from argparse import ArgumentParser
from typing import Tuple, List
import cv2
import numpy as np
from tqdm import tqdm

DEF_SIZES = [640, 1024, 2048]


def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--sizes", type=int, default=DEF_SIZES, nargs="*")
    return parser


def calc_size(orig: Tuple[int, int], big_side: int):
    small_side = int(big_side * (min(orig) / max(orig)))
    big_idx = np.argmax(orig)
    size = (
        big_side if big_idx else small_side,
        big_side if not big_idx else small_side
    )
    return size


def resize_single(in_path: str, out_path: str, size: int):
    if not (os.path.exists(out_path) and os.path.isdir(out_path)):
        raise ValueError(f"{out_path} not exists")
    out = os.path.join(out_path, in_path.split(os.sep)[-1])
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"IMG PROBLEM {out}")
        return None
    size = calc_size(img.shape[:2], size)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out, img)


def resize_multi(
    paths: List[str],
    out_path: str,
    size: int,
    position: int = None
):
    if not (os.path.exists(out_path) and os.path.isdir(out_path)):
        raise ValueError(f"{out_path} not exists")
    for path in tqdm(paths, position=position):
        resize_single(path, out_path, size)


def resize_multi_mp(paths: List[str], out_path: str, size: int):
    if not (os.path.exists(out_path) and os.path.isdir(out_path)):
        os.makedirs(out_path)
    params = [
        (path_list, out_path, size, i)
        for i, path_list in enumerate(np.array_split(paths, os.cpu_count()))
    ]
    with mp.Pool(os.cpu_count()) as pool:
        pool.starmap(resize_multi, params)


if __name__ == "__main__":
    args = create_parser().parse_args()
    photos = [
        os.path.join(args.in_path, filename)
        for filename in os.listdir(args.in_path)
    ]
    for size in args.sizes:
        out_path = os.path.join(args.out_path, f"photos_{size}")
        print(f"Resizing to {size}...")
        resize_multi_mp(photos, out_path, size)
        print("")
