import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join as pjoin

# import config


class PreProcess:
    def __init__(self):
        # BRATS DATASET LOADING SETTINGS
        self.dataset_dir = '/run/media/lzhmark/shared/boe-screen/selected'
        self.images_dir = '{}/{}'.format(self.dataset_dir, 'images')
        # self.images_dir = './origin'
        self.masks_dir = '{}/{}'.format(self.dataset_dir, 'masks')
        self.store_dir = '/run/media/lzhmark/shared/boe-screen/cut'
        self.images_store_dir = '{}/{}'.format(self.store_dir, 'images')
        # self.images_store_dir = './cut'
        self.masks_store_dir = '{}/{}'.format(self.store_dir, 'masks')

    def images_count(self):
        """
        count the number of the images and summarize the sizes of the images
        :return: none
        :returns: print the sizes and the number
        """
        sizes_set = set()
        subdirs = os.listdir(self.images_dir)
        total_length = 0
        with tqdm as pbar:
            for subdir in subdirs:
                image_subdir = pjoin(self.images_dir, subdir)
                image_files = os.listdir(image_subdir)
                for filename in image_files:
                    pbar.update(1)
                    image_filename = pjoin(self.images_dir, subdir, filename)
                    image_file = Image.open(image_filename)
                    if image_file.size not in sizes_set:
                        print('Size:{} filename:{}'.format(image_file.size, image_filename))
                        sizes_set.add(image_file.size)
                    total_length += 1
        print(total_length)

    def images_cut(self):
        """
        cut the images
        :return: save the cut images and masks
        """
        subdirs = os.listdir(self.images_dir)
        with tqdm() as pbar:
            for subdir in subdirs:
                image_subdir = pjoin(self.images_dir, subdir)
                image_files = os.listdir(image_subdir)
                for filename in image_files:
                    pbar.update(1)
                    image_filename = pjoin(self.images_dir, subdir, filename)
                    image_file = Image.open(image_filename)
                    mask_filename = pjoin(self.masks_dir, subdir, '{}.png'.format(filename[:-4]))
                    mask_file = Image.open(mask_filename)
                    if image_file.size == (1224, 1028):
                        cut_img = image_file.crop((0, 0, 1224, 900))
                        cut_mask = mask_file.crop((0, 0, 1224, 900))
                    elif image_file.size == (2464, 2056):
                        cut_img = image_file.crop((0, 0, 2464, 1770))
                        cut_img = cut_img.resize((1224, 900))
                        cut_mask = mask_file.crop((0, 0, 2464, 1770))
                        cut_mask = cut_mask.resize((1224, 900))
                    elif image_file.size == (660, 566):
                        cut_img = image_file.crop((0, 0, 660, 496))
                        cut_img = cut_img.resize((1224, 900))
                    elif image_file.size == (2448, 2056):
                        cut_img = image_file.crop((0, 0, 2448, 1770))
                        cut_img = cut_img.resize((1224, 900))
                    else:
                        raise ValueError('Not occurred size:{}'.format(image_file.size))

                    if not os.path.exists(pjoin(self.images_store_dir, subdir)):
                        os.makedirs(pjoin(self.images_store_dir, subdir))
                    if not os.path.exists(pjoin(self.masks_store_dir, subdir)):
                        os.makedirs(pjoin(self.masks_store_dir, subdir))
                    cut_img.save(pjoin(self.images_store_dir, subdir, filename))
                    cut_mask.save(pjoin(self.masks_store_dir, subdir, filename))


if __name__ == '__main__':
    preprocess = PreProcess()
    preprocess.images_cut()
