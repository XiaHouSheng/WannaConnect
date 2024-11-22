import torch.nn as nn
import torch
import time
from PIL import Image,ImageColor,ImageDraw,ImageChops
import torch.types
from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
import torchvision.transforms as transforms
from torch import optim

#模型
"""识别模型""" #LeNet5
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # 卷积层1：输入通道为1（灰度图像），输出通道为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 池化层1：最大池化，核大小为2x2，步长为2
        self.pool1 = nn.MaxPool2d(2, 1)
        # 卷积层2：输入通道为6，输出通道为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 池化层2：最大池化，核大小为2x2，步长为2
        self.pool2 = nn.MaxPool2d(2, 1)
        # 全连接层1：将卷积层的输出展平后作为输入，输出维度为120
        self.fc1 = nn.Linear(5184, 2048)
        # 全连接层2：输入维度为120，输出维度为84
        self.fc2 = nn.Linear(2048, 1024)
        # 全连接层3：用于分类，输出维度为10（数字0 - 9）
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        #print("1",x.shape)
        x = self.pool1(torch.relu(self.conv1(x)))
        #print("2",x.shape)
        x = self.pool2(torch.relu(self.conv2(x)))
        #print("3",x.shape)
        x = x.view(x.size(0),-1)  # 展平操作
        #print("4",x.shape)
        x = torch.relu(self.fc1(x))
        #print("5",x.shape)
        x = torch.relu(self.fc2(x))
        #print("6",x.shape)
        x = self.fc3(x)
        #print("7",x.shape)
        x = torch.softmax(x,dim=1,dtype=torch.float32)
        return x

"""位置分辨""" #VGG16
class Model2(nn.Module):
    def __init__(self, num_classes=8):
        super(Model2, self).__init__()
        # 特征提取部分（卷积层和池化层）
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四个卷积块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第五个卷积块
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        # 分类器部分（全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
net1 = Model1()
if os.path.exists("./model/model1.pth"):
    net1.load_state_dict(torch.load("./model/model1.pth"))
net2 = Model2()
if os.path.exists("./model/model2.pth"):
    net2.load_state_dict(torch.load("./model/model2.pth"))

myTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0874,),(0.2524,))
])

image = Image.open("./data/test/verify.jpg").convert("L")
image = ImageChops.invert(image)
image = image.resize((120,40))
imageTensor = myTransform(image).unsqueeze(0)
image.close()
outputCut = net2(imageTensor)

imageTest = Image.open("./data/test/verify.jpg").convert("RGB")
imageTest = imageTest.resize()
canvans = ImageDraw.Draw(imageTest)
x1,y1,x2,y2,x3,y3,x4,y4 = tuple([int(i) for i in list(net2(imageTensor)[0])])
canvans.point((x1,y1),(0,256,0))
canvans.point((x2,y2),(0,256,0))
canvans.point((x3,y3),(0,256,0))
canvans.point((x4,y4),(0,256,0))
canvans.rectangle([(x1-14,y1-14),(x1+14,y1+14)],width = 1,outline = (0,256,0))
canvans.rectangle([(x2-14,y2-14),(x2+14,y2+14)],width = 1,outline = (0,256,0))
canvans.rectangle([(x3-14,y3-14),(x3+14,y3+14)],width = 1,outline = (0,256,0))
canvans.rectangle([(x4-14,y4-14),(x4+14,y4+14)],width = 1,outline = (0,256,0))
imageTest.show()            


