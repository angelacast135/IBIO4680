#!/usr/bin/env python3

import numpy as np
import numpy.matlib
import torch
import torchvision
from PIL import Image
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms
import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import pylearn2
import pickle
import pdb

transform = transforms.Compose(
    [transforms.Resize(300),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_train = torchvision.datasets.ImageFolder(root='./data/train_128/', transform=transform)

train_set = torch.utils.data.DataLoader(data_train, batch_size=50,
                                          shuffle=True, num_workers=0)

data_val = torchvision.datasets.ImageFolder(root='./data/val_128/', transform=transform)

val_set = torch.utils.data.DataLoader(data_val, batch_size=50,
                                         shuffle=False, num_workers=0)

data_test = torchvision.datasets.ImageFolder(root='./data/test_128/', transform=transform)

test_set  = torch.utils.data.DataLoader(data_test, batch_size=50,
                                         shuffle=False, num_workers=0)

classes = ('bark1', 'bark2', 'bark3', 'wood1', 'wood2', 'wood3', 'water', 'granite', 'marble', 'floor1', 'floor2', 'pebbles',
           'wall', 'brick1', 'brick2', 'glass1', 'glass2', 'carpet1', 'carpet2', 'upholstery', 'wallpaper', 'fur', 'knit', 'corduroy',
           'plaid')

################################################## Continue to create CNN ######################################## 

from torch.autograd import Variable
import torch.nn as nn
import math
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        print(block.expansion)
        self.fc = nn.Linear(512 * 4 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

model = resnet18(num_classes=25)

print(model)

num_params=0
for p in model.parameters():
    num_params+=p.numel()
print("The number of parameters {}".format(num_params))

#pdb.set_trace()
################################################# Proceeding to train ######################################## 
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import tqdm

criterion = nn.CrossEntropyLoss()
learn_rate = 1e-3
momentumx = 0.915

test_loss_fil = []
train_loss = []


def train(epoch,learn_rate,momentumx,test_accuracy):
	model.train().cuda()
	loss_cum = []
	Acc = 0
	for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_set), total=len(train_set),desc="[TRAIN] Epoch: {}".format(epoch)):
		data = data.cuda(); data = Variable(data)
		target = target.cuda(); target = Variable(target)

		optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum = momentumx)
		optimizer.zero_grad()
		output = model(data)

		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		loss_cum.append(loss.data.cpu()[0])
		_, arg_max_out = torch.max(output.data.cpu(), 1)
		Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
	print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(train_set.dataset)))


#pdb.set_trace()
#################################################################

import os
from os import listdir
from os.path import isfile, join
import glob

imgs=[]


###################################################################
def val(loader, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        loss_cum = []
        Acc = 0
        predictions=[]
        #pdb.set_trace()
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(loader), total=len(loader), desc="[VAL] Epoch: {}".format(epoch)):
#                pdb.set_trace()
                labels = target
                data = data.cuda(); target = target.cuda()
                data, target = Variable(data, volatile=True), Variable(target)

                output = model(data)
                loss = criterion(output,target)

                loss_cum.append(loss.data.cpu()[0])
                test_loss += criterion(output, target).data[0] # sum up batch loss
                _, pred = torch.max(output.data.cpu(),1) # get the index of the max log-probability
                predictions.append(pred)
                _, arg_max_out = torch.max(output.data.cpu(), 1)
                Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
        correct += (pred == labels).sum()

        test_loss /= len(loader.dataset)

        test_accuracy = correct / len(loader.dataset)
        n_loss=np.array(loss_cum).mean()
        n_acc=float(Acc*100)/len(loader.dataset)
        print("Loss Val: %0.3f | Acc Val: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(loader.dataset)))

        return test_accuracy, predictions, n_loss, n_acc




#######################################################################

def test(loader, epoch):
	model.eval()
	test_loss = 0
	correct = 0
	loss_cum = []
	Acc = 0
	predictions=[]
	for batch_idx, (data, target) in tqdm.tqdm(enumerate(loader), total=len(loader), desc="[TEST] Epoch: {}".format(epoch)):  
		labels = target
		data = data.cuda(); target = target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)

		output = model(data)
		loss = criterion(output,target)

		loss_cum.append(loss.data.cpu()[0])
		test_loss += criterion(output, target).data[0] # sum up batch loss
		_, pred = torch.max(output.data.cpu(),1) # get the index of the max log-probability
		predictions.append(pred)
		_, arg_max_out = torch.max(output.data.cpu(), 1)
		Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
	correct += (pred == labels).sum()

	test_loss /= len(loader.dataset)

	test_accuracy = correct / len(loader.dataset)
	n_loss=np.array(loss_cum).mean()
	n_acc=float(Acc*100)/len(loader.dataset)
	print("Loss Test: %0.3f | Acc Test: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(loader.dataset)))

	return test_accuracy, predictions,n_loss,n_acc

#pdb.set_trace()

print('Train Accuracy')

num_epoch = 50
test_accuracy = 0
v_loss=[]
v_acc=[]
for epoch in range(1, num_epoch):
    train(epoch,learn_rate,momentumx,test_accuracy)

    t_a1, predictions2, n_loss,n_acc = val(val_set, epoch)

    t_a, predictions, n_loss2,n_acc2 = test(test_set, epoch)
    v_loss.append(n_loss)
    v_acc.append(n_acc)

    torch.save(model.state_dict(), './best_model_03')
    print('saved model')


with open('./v_loss.pickle', 'wb') as lss:
            pickle.dump(v_loss, lss, protocol=pickle.HIGHEST_PROTOCOL)
with open('./v_acc.pickle', 'wb') as accc:
            pickle.dump(v_acc, accc, protocol=pickle.HIGHEST_PROTOCOL)

