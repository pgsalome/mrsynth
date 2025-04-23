import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool

def image_write(path_A, path_B, path_C, path_ABC):
    im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
    im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
    im_C = cv2.imread(path_C, cv2.IMREAD_COLOR)
    im_ABC = np.concatenate([im_A, im_B, im_C], axis=1)
    im_ABC_resized = cv2.resize(im_ABC, (720, 240))  # Resize the final image to 480x240
    cv2.imwrite(path_ABC, im_ABC_resized)

parser = argparse.ArgumentParser(description='create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, required=True)
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, required=True)
parser.add_argument('--fold_C', dest='fold_C', help='input directory for image C', type=str, required=True)
parser.add_argument('--fold_ABC', dest='fold_ABC', help='output directory', type=str, required=True)
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_ABC', dest='use_ABC', help='if true: (0001_A, 0001_B, 0001_C) to (0001_ABC)', action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If used, chooses single CPU execution instead of parallel execution', action='store_true', default=False)
args = parser.parse_args()

for arg in vars(args):
    print(f'[{arg}] = {getattr(args, arg)}')

splits = os.listdir(args.fold_A)

if not args.no_multiprocessing:
    pool = Pool()

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_fold_C = os.path.join(args.fold_C, sp)
    img_list = os.listdir(img_fold_A)
    if args.use_ABC:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print(f'split = {sp}, use {num_imgs}/{len(img_list)} images')
    img_fold_ABC = os.path.join(args.fold_ABC, sp)
    if not os.path.isdir(img_fold_ABC):
        os.makedirs(img_fold_ABC)
    print(f'split = {sp}, number of images = {num_imgs}')
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        if args.use_ABC:
            name_B = name_A.replace('_A.', '_B.')
            name_C = name_A.replace('_A.', '_C.')
        else:
            name_B = name_A
            name_C = name_A
        path_B = os.path.join(img_fold_B, name_B)
        path_C = os.path.join(img_fold_C, name_C)
        if os.path.isfile(path_A) and os.path.isfile(path_B) and os.path.isfile(path_C):
            name_ABC = name_A
            if args.use_ABC:
                name_ABC = name_ABC.replace('_A.', '.')  # remove _A
            path_ABC = os.path.join(img_fold_ABC, name_ABC)
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B, path_C, path_ABC))
            else:
                im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
                im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
                im_C = cv2.imread(path_C, cv2.IMREAD_COLOR)
                im_ABC = np.concatenate([im_A, im_B, im_C], axis=1)
                im_ABC_resized = cv2.resize(im_ABC, (720, 240))  # Resize the final image to 480x240
                cv2.imwrite(path_ABC, im_ABC_resized)

if not args.no_multiprocessing:
    pool.close()
    pool.join()
