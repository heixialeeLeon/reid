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

parser = argparse.ArgumentParser(description="train Person ReID model")

# training parameters
parser.add_argument("--image_size", type=int, default=256,help="training patch size")
parser.add_argument("--batch_size", type=int, default=32,help="batch size")
parser.add_argument("--epochs", type=int, default=60,help="number of epochs")
parser.add_argument("--save_per_epoch", type=int, default=5,help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--steps_show", type=int, default=100,help="steps per epoch")
parser.add_argument("--output_path", type=str, default="checkpoints",help="checkpoint dir")
parser.add_argument("--devices", type=str, default="cuda:0",help="device description")
parser.add_argument("--resume_model", type=str, default=None, help="resume model path")

args = parser.parse_args()
print(args)

def save_model(model,epoch):
    '''save model for eval'''
    ckpt_name = '/reid_epoch_{}.pth'.format(epoch)
    path = args.output_path
    if not os.path.exists(path):
        os.mkdir(path)
    path_final = path + ckpt_name
    print('Saving checkpoint to: {}\n'.format(path_final))
    torch.save(model.state_dict(), path_final)

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

def train_model(model, criterion,optimizer,scheduler, num_epochs):
    for epoch in range(num_epochs):
        print('------------------------------------------------')
        epoch_start_time = time.time()
        running_train_loss = 0.0
        running_train_corrects = 0.0
        running_val_loss = 0.0
        running_val_corrects = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                data_loader = train_loader
                model.train(True)

                for index, data in enumerate(data_loader):
                    inputs, labels = data
                    batch_size, c, h, w = inputs.shape
                    # if (batch_size < args.batch_size):
                    #     continue
                    inputs = tensor_to_device(inputs)
                    labels = tensor_to_device(labels)
                    optimizer.zero_grad()
                    outputs = model(Variable(inputs))
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    if index % args.steps_show == 0:
                        print("{}/{} current loss : {}".format(index, len(data_loader), loss.item()))
                    # statistics
                    running_train_loss += loss.item()*batch_size
                    running_train_corrects += float(torch.sum(preds == labels.data))
            else:
                data_loader = val_loader
                model.train(False)
                for data in data_loader:
                    inputs, labels = data
                    batch_size, c, h, w = inputs.shape
                    # if (batch_size < args.batch_size):
                    #     continue
                    inputs = tensor_to_device(inputs)
                    labels = tensor_to_device(labels)
                    optimizer.zero_grad()

                    with torch.no_grad():
                        outputs = model(Variable(inputs))
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # statistics
                    running_val_loss += loss.item()*batch_size
                    running_val_corrects += float(torch.sum(preds == labels.data))

        epoch_train_loss = running_train_loss / len(image_datasets['train'])
        epoch_train_acc = running_train_corrects / len(image_datasets['train'])
        epoch_val_loss = running_val_loss / len(image_datasets['val'])
        epoch_val_correct = running_val_corrects / len(image_datasets['val'])

        time_elapsed = time.time() - epoch_start_time
        print("Epoch {}/{} cost {:.0f}m {:.0f}s".format(epoch, num_epochs - 1,time_elapsed // 60, time_elapsed % 60))
        print('Train Loss: {:.4f} Acc: {:.4f}  Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_correct))

        if (epoch + 1) % args.save_per_epoch == 0:
            save_model(model, epoch + 1)


# prepare the dataloader
train_loader = DataLoader(image_datasets['train'],batch_size=args.batch_size,shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(image_datasets['val'],batch_size=args.batch_size,shuffle=False, num_workers=8, pin_memory=True)
class_names = image_datasets['train'].classes
#print(class_names)

# prepare the model
model = ft_net(len(class_names))
model = model_to_device(model)
print(model)

# prepare the criterion
criterion = nn.CrossEntropyLoss()

# prepare the optimizer
parameters = model.parameters()
classisfier_parameters = model.classifier.parameters()
ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params':base_params, 'lr': 0.1*args.lr},
    {'params': model.classifier.parameters(), 'lr':args.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# prepare the scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

train_model(model,criterion,optimizer_ft,exp_lr_scheduler, args.epochs)