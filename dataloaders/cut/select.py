import os
from shutil import copy, move


def f1():
    """
    考虑到现有数据集
    该方法能够：
    从mask和image中，根据文件名找到互相对应的两张图片
    并一起拷贝到selected文件夹下
    忽略找不到对应的image和mask图片和找不到对应mask的image图片
    """
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


def f2():
    """
    该函数访问images文件夹中所有子文件夹下所有文件
    若masks文件夹下同名子文件夹不包含同名的mask，则将image移动到single文件下下同名子文件夹
    """
    print("开始选择")
    path = '/run/media/lzhmark/shared/boe-screen/all/'
    path_image = os.path.join(path, 'images')
    path_mask = os.path.join(path, 'masks')
    path_single = os.path.join(path, 'singles')

    size, count = 0, 0
    # for each dir and sub-dir
    for root, _, files in os.walk(os.path.join(path_image)):
        # for each file in dir
        for file in files:
            size += 1
            image_path = os.path.join(root, file)
            mask_path = image_path.replace('images', 'masks').replace('.jpg', '.png')
            if not os.path.exists(mask_path):
                single_path = image_path.replace('images', 'singles')
                if not os.path.exists(os.path.dirname(single_path)):
                    os.makedirs(os.path.dirname(single_path))
                move(image_path, single_path)
                count += 1
    print("从{}张image中找到{}张孤立的image".format(size, count))


def f3():
    """
    找出image中没有mask的和mask中没有image的
    """
    print("开始选择")
    path = '/run/media/lzhmark/shared/boe-screen/'
    path_image = os.path.join(path, 'all', 'images')

    diff = {"extra-image": [], "extra-mask": []}
    for root, _, files in os.walk(os.path.join(path_image)):
        if os.path.split(root)[0] == path_image:
            images = os.listdir(root)
            masks = os.listdir(root.replace('images', 'masks'))
            _class = os.path.basename(root)
            if len(images) != len(masks):
                for image in images:
                    if image.replace('.jpg', '.png') not in masks:
                        diff['extra-image'].append((image, _class))
                for mask in masks:
                    if mask.replace('.png', '.jpg') not in images:
                        diff['extra-mask'].append((mask, _class))
    print(diff)


if __name__ == '__main__':
    # f1()
    # f2()
    # f3()
    pass
