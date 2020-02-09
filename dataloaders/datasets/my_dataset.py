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
    def __init__(self, file_names):
        self.mean = (0.5071, 0.4867, 0.4408)
        self.stdv = (0.2675, 0.2565, 0.2761)
        self.dir = Path.db_root_dir("mydataset")
        # {'true':[name1,name2],'false':[name3,name4]}
        files = []
        for _class in ['true', 'false']:
            for file in file_names[_class]:
                files.append([file, _class])
        self.files = sorted(files)

    def __getitem__(self, idx):
        _class = self.files[idx][1]
        img = Image.open(os.path.join(self.dir, 'images', _class, self.files[idx][0] + '.jpg')).convert('RGB')
        mask = Image.open(os.path.join(self.dir, 'masks', _class, self.files[idx][0] + '.jpg'))
        mask = np.array(mask, dtype=np.uint8)
        mask = self._decode(mask, _class)
        target = Image.fromarray(mask)
        sample = {'image': img, 'label': target}
        return self.transform(sample)

    def __len__(self):
        return len(self.files)

    def transform(self, sample):
        train_transform = transforms.Compose([
            # tr.FixedResize(700),  # 如果内存不够就resize
            # tr.Normalize(mean=self.mean, std=self.stdv),
            tr.ToTensor(),
        ])
        return train_transform(sample)

    def _decode(self, mask, _class):
        # _class为false时，将mask中非背景的部分变为2
        if _class == 'true':
            mask[mask != 0] = 1
        else:
            mask[mask != 0] = 1  # 2,不分类
        return mask

    @staticmethod
    def apart(test_size):
        """
        test_size：从每个class值中取多少张作为测试集
        """
        assert 0 < test_size < 1, "比例啦"

        # 读取文件名
        file_names_true, file_names_false = [], []
        dir = Path.db_root_dir("mydataset")
        for _class in ['true', 'false']:
            for root, __, files in os.walk(os.path.join(dir, 'images', _class)):
                for file in files:
                    if _class == 'true':
                        file_names_true.append(os.path.splitext(file)[0])
                    else:
                        file_names_false.append(os.path.splitext(file)[0])

        # debug
        # file_names_true = file_names_true[:8]
        # file_names_false = file_names_false[:8]

        def rand_select(data, size):
            test = []
            index = [i for i in range(len(data))]
            idx = random.sample(index, size)
            for i in idx:
                test.append(data[i])
            train = [data[i] for i in index if i not in idx]
            return train, test

        train_data_true, test_data_true = rand_select(file_names_true, 1 + int(len(file_names_true) * test_size))
        train_data_false, test_data_false = rand_select(file_names_false, 1 + int(len(file_names_false) * test_size))

        train_inputs_true, train_inputs_false, test_inputs_true, test_inputs_false = [], [], [], []
        for train_img in train_data_true:
            train_inputs_true.append(str(train_img))
        for test_img in test_data_true:
            test_inputs_true.append(str(test_img))
        for train_img in train_data_false:
            train_inputs_false.append(str(train_img))
        for test_img in test_data_false:
            test_inputs_false.append(str(test_img))

        train_inputs = {'true': train_inputs_true, 'false': train_inputs_false}
        test_inputs = {'true': test_inputs_true, 'false': test_inputs_false}

        return train_inputs, test_inputs


if __name__ == '__main__':
    """
    test dataloader
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import torchvision
    from dataloaders.utils import decode_segmap

    train_files, _ = MyDataset.apart(0.1)
    dataset = MyDataset(train_files)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    samples = next(iter(train_loader))


    def printer(images, type):
        if type == 'label':
            for idx, label_mask in enumerate(images):
                plt.title('masks{}'.format(idx))
                plt.imshow(decode_segmap(np.array(label_mask), 'mydataset'))
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
