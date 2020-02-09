import os
import numpy
from shutil import copy

"""
考虑到现有数据集
该方法能够：
从mask和image中，根据文件名找到互相对应的两张图片
并一起拷贝到selected文件夹下
忽略找不到对应的image和mask图片和找不到对应mask的image图片
"""
if __name__ == '__main__':
    print("开始选择")
    path = '/run/media/lzhmark/shared/boe-screen'
    path_mask, path_image = os.path.join(path, 'origin', 'masks'), os.path.join(path, 'origin', 'images')
    path_mask_true, path_mask_false = os.path.join(path_mask, 'true'), os.path.join(path_mask, 'false')
    path_image_true, path_image_false = os.path.join(path_image, 'true'), os.path.join(path_image, 'false')

    # selected目录
    path = os.path.join(path, 'selected')
    path_false_masks, path_true_masks = os.path.join(path, 'masks', 'false'), os.path.join(path, 'masks', 'true')
    path_false_images, path_true_images = os.path.join(path, 'images', 'false'), os.path.join(path, 'images', 'true')
    for p in [path, path_false_masks, path_true_masks, path_false_images, path_true_images]:
        if not os.path.exists(p):
            os.makedirs(p)

    # 读取所有的文件名
    masks_true, masks_false, images_true, images_false = [], [], [], []
    for i, file_dir in enumerate([path_mask_false, path_mask_true, path_image_false, path_image_true]):
        for root, __, files in os.walk(file_dir):
            for image in files:
                if i == 0:
                    masks_false.append(os.path.splitext(image)[0])
                elif i == 1:
                    masks_true.append(os.path.splitext(image)[0])
                elif i == 2:
                    images_false.append(os.path.splitext(image)[0])
                elif i == 3:
                    images_true.append(os.path.splitext(image)[0])

    masks_false = numpy.array(masks_false)
    masks_true = numpy.array(masks_true)
    images_false = numpy.array(images_false)
    images_true = numpy.array(images_true)
    print("开始拷贝")

    for _class in [True, False]:
        count = 0
        print("COPYING {}".format(_class))
        for mask in masks_true if _class else masks_false:
            for image in images_true if _class else images_false:
                if mask == image:
                    # copy mask
                    src = os.path.join(path_mask_true if _class else path_mask_false, "{}.png".format(mask))
                    copy(src, path_true_masks if _class else path_false_masks)
                    # copy image
                    src = os.path.join(path_image_true if _class else path_image_false, "{}.jpg".format(image))
                    copy(src, path_true_images if _class else path_false_images)
                    count += 1

        print("{}从{}张mask和{}张image中找到{}组".format(_class, len(masks_true if _class else masks_false),
                                                 len(images_true if _class else images_false), count))
    exit(0)
