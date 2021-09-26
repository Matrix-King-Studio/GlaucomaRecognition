import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

if __name__ == '__main__':
    data_path = "Datasets/Training400"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强和标准化
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in ['train', 'val']}
    dataloader = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(class_names)
