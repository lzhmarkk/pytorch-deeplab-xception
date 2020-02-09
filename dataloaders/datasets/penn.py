import torch
import os
import numpy as np
from PIL import Image
from dataloaders import custom_transforms as tr
from torchvision import transforms
from torch.utils.data import Dataset
from mypath import Path


class PennFudanDataset(Dataset):
    def __init__(self):
        self.root = Path.db_root_dir('penn')
        self.transform = transforms.Compose([
            tr.FixedResize(550),
            tr.ToTensor()
        ])
        self.PNGImagesRoot = "PNGImages"
        self.PedMasksRoot = "PedMasks"
        # 加载image和mask，sort是为了保证他们一一对应
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, self.PNGImagesRoot))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root, self.PedMasksRoot))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.PNGImagesRoot, self.imgs[idx])
        mask_path = os.path.join(self.root, self.PedMasksRoot, self.masks[idx])

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)  # 因为颜色取决于id，我们就不给他上色了这里

        # 转化成numpy的array
        mask = np.array(mask)
        obj_ids = np.unique(mask)  # 赋予id，也就是颜色
        obj_ids = obj_ids[1:]  # obj_id[0]是背景

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]  # align id to mask(据观察大概是masks[k][i]=True if mask[i]==ids[k] for k)
        # 得到外框
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])  # find the k'th box's i
            # 这里，pos就是在img中第i大的元素的位置list，即围成第i个mask所有的点的坐标
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 把所有数据转化成tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # 面积
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # 暂时没有crowd

        full_mask = torch.zeros(mask.shape)
        for m in masks:
            full_mask[m != 0] = 1  # 全部给上红色
        mask = Image.fromarray(np.array(full_mask))
        sample = {'image': img, 'label': mask}
        return self.transform(sample)

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    """
    test dataloader
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import torchvision
    from dataloaders.utils import decode_segmap

    dataset = PennFudanDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)


    def printer(images, type):
        if type == 'label':
            for idx, label_mask in enumerate(images):
                plt.title('masks{}'.format(idx))
                plt.imshow(decode_segmap(np.array(label_mask), 'penn'))
                plt.show()
        elif type == 'image':
            sample = torchvision.utils.make_grid(images, normalize=True)
            sample = sample.numpy().transpose((1, 2, 0))
            plt.title('images')
            plt.imshow(sample)
            plt.show()
        elif type == 'mix':
            for idx, label in enumerate(images['label']):
                plt.title('mix{}'.format(idx))
                mask = decode_segmap(np.array(label), 'penn')
                image = torchvision.utils.make_grid(images['image'][idx], normalize=True)
                image = image.numpy().transpose((1, 2, 0))
                plt.title('images')
                plt.imshow(mask + image)
                plt.show()


    for epoch, samples in enumerate(loader):
        # printer(samples['image'], type='image')
        # printer(samples['label'], type='label')
        printer(samples, type='mix')
        if epoch > 3:
            break
    exit(0)
