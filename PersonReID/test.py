import torch
import torch.nn as nn
from torch.optim import Adam, SGD,lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from model.model import *
import argparse
from dataset import *
import time
import matplotlib
import matplotlib.pyplot as plt
import scipy.io

parser = argparse.ArgumentParser(description="train Person ReID model")
# training parameters
parser.add_argument("--test_dir", type=str, default="/data_1/data/REID/Market-1501-v15.09.15/pytorch", help="resume model path")
parser.add_argument("--image_size", type=int, default=256,help="training patch size")
parser.add_argument("--batch_size", type=int, default=32,help="batch size")
parser.add_argument("--devices", type=str, default="cuda:0",help="device description")
parser.add_argument("--resume_model", type=str, default='checkpoints/reid_epoch_60.pth', help="resume model path")

args = parser.parse_args()
print(args)

def resume_model(model, model_path):
    print("Resume model from {}".format(args.resume_model))
    model.load_state_dict(torch.load(model_path))

def model_to_device(model):
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def tensor_to_device(tensor):
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    return tensor.to(device)

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            #if opt.fp16:
            #    input_img = input_img.half()
            outputs = model(input_img)
            f = outputs.data.cpu().float()
            ff = ff+f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff), 0)
    return features

data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {x: datasets.ImageFolder(os.path.join(args.test_dir,x),data_transforms) for x in ['gallery', 'query']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size,shuffle=False, num_workers=4) for x in ['gallery', 'query']}

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

# prepare the model
model = ft_net(751)
resume_model(model,args.resume_model)
model.classifier.classifier = nn.Sequential()  # remove the last fc
model = model.eval()
model = model_to_device(model)

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])

print("gallery feature shape: {}".format(gallery_feature.shape))
print("query feature shape: {}".format(query_feature.shape))

result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)