# -*- coding:utf-8 -*-
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import os
from dataloaders import custom_transforms as tr
from mypath import Path


class MyDataset(Dataset):
    def __init__(self, file_names, dataset, classify):
        self.mean = (0.5071, 0.4867, 0.4408)
        self.stdv = (0.2675, 0.2565, 0.2761)
        self.dir = Path.db_root_dir(dataset)
        self.dataset = dataset
        self.classify = classify
        print("本次运行{}分类".format('使用' if self.classify == 'True' else '只分割不'))
        files = []
        self.classes = list(file_names.keys())
        for _class in self.classes:
            for file in file_names[_class]:
                files.append([file, _class])
        self.files = sorted(files)

    def __getitem__(self, idx):
        _class = self.files[idx][1]
        ext = '.png' if self.dataset == 'all' else '.jpg'
        img = Image.open(os.path.join(self.dir, 'images', _class, self.files[idx][0] + '.jpg')).convert('RGB')
        mask = Image.open(os.path.join(self.dir, 'masks', _class, self.files[idx][0] + ext))
        mask = mask.resize(img.size)  # forced resize
        mask = np.array(mask, dtype=np.uint8)
        mask = self._decode(mask, _class)
        target = Image.fromarray(mask)
        sample = {'image': img, 'label': target}
        return self.transform(sample)

    def __len__(self):
        return len(self.files)

    def transform(self, sample):
        train_transform = transforms.Compose([
            tr.FixedResize(700),  # 如果内存不够就resize
            # tr.Normalize(mean=self.mean, std=self.stdv),
            tr.ToTensor(),
        ])
        return train_transform(sample)

    def _decode(self, mask, _class):
        # 若不做分类(self.classify=False)，则把所有的都视为同一类
        if self.classify == 'True':
            mask[mask != 0] = self.classes.index(_class) + 1
        else:
            mask[mask != 0] = 1
        return mask

    @staticmethod
    def apart(test_size, dataset):
        """
        test_size：从每个class值中取多少张作为测试集
        """
        assert 0 < test_size < 1, "比例啦"

        # 读取文件名
        file_names = {}
        classes = []
        dir = Path.db_root_dir(dataset)
        for root, dirs, files in os.walk(os.path.join(dir, 'masks')):
            # 如果是/mask的一级子文件夹
            if os.path.split(root)[0] == os.path.join(dir, 'masks'):
                _class = os.path.split(root)[1]  # 子文件夹名
                classes.append(_class)
                for file in files:
                    if _class not in file_names.keys():
                        file_names[_class] = []
                    file_names[_class].append(os.path.splitext(file)[0])

        def rand_select(data, size):
            test = []
            index = [i for i in range(len(data))]
            idx = random.sample(index, size)
            for i in idx:
                test.append(data[i])
            train = [data[i] for i in index if i not in idx]
            return train, test

        train_data, test_data = {}, {}
        for _class in classes:
            train_data[_class], test_data[_class] = rand_select(file_names[_class],
                                                                int(len(file_names[_class]) * test_size))
            print("{}：训练{}张，测试{}张".format(_class, len(train_data[_class]), len(test_data[_class])))

        return train_data, test_data


if __name__ == '__main__':
    """
    test dataloader
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import torchvision
    from dataloaders.utils import decode_segmap

    _dataset = 'all'

    train_files, _ = MyDataset.apart(0.1, _dataset)
    dataset = MyDataset(train_files, dataset=_dataset, classify=False)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    samples = next(iter(train_loader))


    def printer(images, type):
        if type == 'label':
            for idx, label_mask in enumerate(images):
                plt.title('masks{}'.format(idx))
                plt.imshow(decode_segmap(np.array(label_mask), _dataset))
                plt.show()
        else:
            sample = torchvision.utils.make_grid(images, normalize=True)
            sample = sample.numpy().transpose((1, 2, 0))
            plt.title('images')
            plt.imshow(sample)
            plt.show()


    printer(samples['image'], type='image')
    printer(samples['label'], type='label')
    exit(0)
