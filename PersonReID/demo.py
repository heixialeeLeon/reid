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
from transform.image_show import *
from PIL import Image, ImageFont, ImageDraw

parser = argparse.ArgumentParser(description="train Person ReID model")
# training parameters
parser.add_argument("--test_dir", type=str, default="/data_1/data/REID/Market-1501-v15.09.15/pytorch", help="resume model path")
parser.add_argument("--image_size", type=int, default=256,help="training patch size")
parser.add_argument("--batch_size", type=int, default=32,help="batch size")
parser.add_argument("--devices", type=str, default="cuda:0",help="device description")
parser.add_argument("--resume_model", type=str, default='checkpoints/reid_epoch_60.pth', help="resume model path")

args = parser.parse_args()
print(args)

def sort_img(f,f_label,f_cam,gallery_features, gallery_labels, gallery_cams):
    f = f.view(-1,1)
    score = torch.mm(gallery_features,f)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    #predict index
    index = np.argsort(score)
    index = index[::-1]
    # good index
    query_index  = np.argwhere(gallery_labels == f_label)
    # same camera
    camera_index = np.argwhere(gallery_cams == f_cam)

    junk_index1 = np.argwhere(gallery_label == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2,junk_index1)

    mask = np.in1d(index ,junk_index,invert=True)
    index = index[mask]
    return index

def sort_img2(f,f_label,f_cam,gallery_features, gallery_labels, gallery_cams):
    f = f.view(-1, 1)
    score = torch.mm(gallery_features, f)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)
    index = index[::-1]

    camera_index = np.argwhere(gallery_cams == f_cam)
    mask = np.in1d(index, camera_index, invert = True)
    index = index[mask]
    return index

def find_max_match(query_index,match_num=10):
    index = sort_img2(query_feature[query_index], query_label[query_index], query_cam[query_index], gallery_feature, gallery_label, gallery_cam)
    show_imgs = []
    query_img_path, _ = image_datasets['query'].imgs[query_index]
    query_img = Image.open(query_img_path).convert('RGB')
    show_imgs.append(query_img)
    # CV2_showPILImage(query_img)
    for i in range(match_num):
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        img = Image.open(img_path).convert('RGB')
        show_imgs.append(img)
    CV2_showPILImage_List(show_imgs,timeout=3000)

# prepare the dataset
image_datasets = {x: datasets.ImageFolder( os.path.join(args.test_dir,x) ) for x in ['gallery','query']}

# load the result
result = scipy.io.loadmat("pytorch_result.mat")
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

# find the index with the hight similarity
# index = sort_img(query_feature[query_index],query_label[query_index],query_cam[query_index],gallery_feature,gallery_label,gallery_cam)
#
# show_imgs = []
# query_img_path, _= image_datasets['query'].imgs[query_index]
# query_img = Image.open(query_img_path).convert('RGB')
# show_imgs.append(query_img)
# # CV2_showPILImage(query_img)
# for i in range(10):
#     img_path, _ = image_datasets['gallery'].imgs[index[i]]
#     img = Image.open(img_path).convert('RGB')
#     show_imgs.append(img)
for index in range(len(image_datasets['query'])):
    find_max_match(index)