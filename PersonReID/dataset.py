import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from shutil import copyfile
import os
from  transform.image_show import *

version = torch.__version__

data_dir = "/data_1/data/REID/Market-1501-v15.09.15/pytorch"

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),data_transforms['val'])

# for the dataset
# if __name__ == "__main__":
#     for item in image_datasets['train']:
#         print(item[0].shape)
#         print(item[1])

# for the dataloader
if __name__ == "__main__":
    loader = DataLoader(image_datasets['train'],batch_size=1,shuffle=False)
    for item in loader:
        # print(item[0].shape)
        # print(item[1].shape)
        CV2_showTensors(item[0])