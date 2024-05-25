import os

import torch
from PIL import Image


class WSIClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root, type='train', transform=None):
        self.root = root
        self.type = type
        self.transform = transform
        self.classes = ['出血', '坏死', '实质', '淋巴', '空泡', '空白', '间质']
        # 加载所有图片文件，并对文件进行排序
        self.images_info_file = os.path.join(self.root, 'meta', f'{self.type}.txt')
        self.images, self.labels = [], []
        with open(self.images_info_file, 'r', encoding='utf-8') as f:
            for line in f:
                path, label = line.split(' ')
                self.images.append(os.path.join(self.root, self.type, path))
                self.labels.append(int(label))

    def __getitem__(self, idx):
        # 加载图片以及标签
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    classes = ['出血', '坏死', '实质', '淋巴', '空泡', '空白', '间质']
    num_classes = len(classes)
    print(len(classes))

    root = '../../datasets/Classification'
    dataset = WSIClassificationDataset(root)
    print(dataset[1256])
    x, y = dataset[1256]
    x.show()
    print(y)
