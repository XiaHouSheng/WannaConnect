import torch.nn as nn
import torch
import time
from PIL import Image,ImageColor,ImageDraw,ImageFont
import torch.types
from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
import torchvision.transforms as transforms
from torch import optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

#拼接函数|预处理
# Target 60x20 3:1 画布设置 120*40
def splicingDataImage(images:list,name = "") -> dict:
    imageCanvans = Image.new("L",[120,40],0)
    labels = ""
    points = []
    for index in range(4):
        image_name = images[index]
        labels += str((image_name.split(".")[0]).split("_")[2])
        image = Image.open("./data/train/{}".format(image_name))
        noise = random.random() #位移噪点
        pasteY = int(noise * 18)
        pasteX = 28 * (index) + int(noise * 5)
        points.append([str(pasteX+14),str(pasteY+14)])
        imageCanvans.paste(image,(pasteX,pasteY))
        #draw = ImageDraw.Draw(imageCanvans)
        #draw.point((pasteX+14,pasteY+14),256)
    #imageCanvans.show()
    part_points = ",".join(["{0}#{1}".format(i[0],i[1]) for i in points])
    #training_123|123,123|123,123|123,123|123_4170.png
    finalName = "_".join(["training",name,part_points,labels]) + ".png"
    imageCanvans.save("./data/final_train/{}".format(finalName))
    imageCanvans.close()
    print("Save {} Success!".format(finalName))

#字体生成非手写
def generateImageDigital(name = ""):
    points = []
    labels = [str(random.randint(0,9)) for i in range(4)]
    label = "".join(labels)
    image = Image.new("L",(120,40),255)
    canvans = ImageDraw.Draw(image)
    font = ImageFont.truetype("./font/font.otf", 20)
    gaussian_noise = np.random.normal(0,25,(40,120))
    for i in range(4):
        pasteX = 30 * i + random.randint(2,20)
        pasteY = random.randint(0,20)
        points.append([str(pasteX+3),str(pasteY+10)])
        canvans.text((pasteX,pasteY),str(labels[i]),0,font = font)
        #canvans.point((pasteX+3,pasteY+10),256)
    image = image + gaussian_noise
    image = Image.fromarray(image).convert("L")
    part_points = ",".join(["{0}#{1}".format(i[0],i[1]) for i in points])
    #training_123|123,123|123,123|123,123|123_4170.png
    finalName = "_".join(["training",name,part_points,label]) + ".png"
    #image.show()
    image.save("./data/final_train/{}".format(finalName))
    image.close()
    
"""生成样本"""
def generateImages():
    for i in range(999,6001):
        generateImageDigital(str(i))

#数据导入|标准化
batch_size = 16
batch_size_detect = 32
height = 40
width =120
path_log = "./logs/"

myTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.9019,),(0.2204,))
])

myDetectTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1337,),(0.3021,))
])

class MyDataSet(Dataset):
    def __init__(self):
        self.data = os.listdir("./data/final_train/")

    def __getitem__(self,index) -> tuple:
        name = self.data[index]
        path = "./data/final_train/{}".format(name)
        image = Image.open(path).convert("L")
        image = myTransform(image)
        positions = [[int(a) for a in i.split("#")] for i in name.split("_")[2].split(",")]
        positions = [element for a in positions for element in a]
        positions = torch.tensor(positions,dtype=torch.float32)
        label = torch.tensor([int(i) for i in name.split(".")[0].split("_")[-1]],dtype=torch.float32)
        return image,positions,label,name

    def __len__(self):
        return len(self.data)

class MyDetectDataSet(Dataset):
    def __init__(self):
        self.data = os.listdir("./data/train/")

    def __getitem__(self,index) -> tuple:
        name = self.data[index]
        path = "./data/train/{}".format(name)
        image = Image.open(path).convert("L")
        image = myDetectTransform(image)
        index = int(name.split(".")[0].split("_")[2])
        label = [0 for i in range(10)]
        label[index-1] = 1
        label = torch.tensor(label,dtype=torch.float32)
        return image,label,name

    def __len__(self):
        return len(self.data)
    
myDataSet = MyDataSet() #分割模型
myDataLoader = DataLoader(myDataSet,batch_size=batch_size,shuffle=True) #分割模型
myTestDataLoader = DataLoader(myDataSet,batch_size=1,shuffle=True) #分割模型

myDetectDataSet = MyDetectDataSet() #识别模型
myDetectDataLoader = DataLoader(myDetectDataSet,batch_size=batch_size_detect,shuffle=True) #识别模型
myDetectTestDataLoader = DataLoader(myDetectDataSet,batch_size=1,shuffle=True) #识别模型

#标准化参数mean和std计算
def getMeanAStd():
    num_data = len(myDataSet)
    sum = torch.zeros(1)
    squared_sum = torch.zeros(1)
    for index in range(num_data):
        image,a,b,c = myDataSet[index]
        image = transforms.ToTensor()(image)
        sum += torch.sum(image)
        squared_sum += torch.sum(image ** 2)
    total_pixels = 120 * 40 * num_data
    mean = sum / total_pixels
    std = torch.sqrt(squared_sum / total_pixels - mean**2)
    print("mean",mean,"std",std)

# mean 0.9019 std 0.2204 分割模型
# mean 0.1337 std 0.3021 识别模型
   
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

"""位置分辨""" #ResNet50
class Model2(nn.Module):
    def __init__(self, num_classes=8):
        super(Model2, self).__init__()
        # 特征提取部分（卷积层和池化层）
        self.features = models.resnet50(pretrained=True)
        """改成单通道"""
        self.features.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 分类器部分（全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
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

#训练优化
lossModel2 = nn.MSELoss()
optimizerModel2 = optim.Adam(net2.parameters(),lr = 0.0001)

lossModel1 = nn.MSELoss()
optimizerModel1 = optim.Adam(net1.parameters(),lr = 0.0001)

def trainCutModel(epoch=5):
    path = path_log+"trainCutModel_{}".format(str(int(time.time())))
    os.mkdir(path)
    writer = SummaryWriter(path)
    for i in range(epoch):
        net2.train()
        epoch_loss = 0.0
        for index,(images,positions,_,_) in enumerate(myDataLoader):
            outputs = net2(images)
            #print("positions",positions.shape,"outputs",outputs.shape)
            loss = lossModel2(outputs,positions)
            optimizerModel2.zero_grad()
            loss.backward()
            optimizerModel2.step()
            """上传损失"""
            writer.add_scalar("CutModel/Train/Epoch"+str((i+1)),loss.item(),index)
            #print("epoch",i,"index",index,"loss",loss.item())
            epoch_loss += loss.item()    
        image = testCutModel()
        """每一个epoch进行一次测试"""
        writer.add_image("CutModel/Test/Epoch"+str((i+1)),np.array(image),i+1,dataformats="HWC")
        print("epoch",i+1,"loss",epoch_loss / int(5000/batch_size))
        print("-----------------------------")
        epoch_loss = 0.0
    writer.close()
    
    torch.save(net2.state_dict(),'./model/model2.pth')
    print("Save PTH Model2 Success!")

def testCutModel(show = False):
    net2.eval()
    for index,(images,_,_,name) in enumerate(myTestDataLoader):
            image = Image.open("./data/final_train/{}".format(name[0])).convert("RGB")
            positions = [[int(a) for a in i.split("#")] for i in name[0].split("_")[2].split(",")]
            canvas = ImageDraw.Draw(image)
            x1,y1,x2,y2,x3,y3,x4,y4 = tuple([int(i) for i in list(net2(images)[0])])
            canvas.point((x1,y1),(0,256,0))
            canvas.point((x2,y2),(0,256,0))
            canvas.point((x3,y3),(0,256,0))
            canvas.point((x4,y4),(0,256,0))
            canvas.rectangle([(x1 - 10, y1 - 10), (x1 + 10, y1 + 10)], width = 1, outline = (0, 256, 0))
            canvas.rectangle([(x2 - 10, y2 - 10), (x2 + 10, y2 + 10)], width = 1, outline = (0, 256, 0))
            canvas.rectangle([(x3 - 10, y3 - 10), (x3 + 10, y3 + 10)], width = 1, outline = (0, 256, 0))
            canvas.rectangle([(x4 - 10, y4 - 10), (x4 + 10, y4 + 10)], width = 1, outline = (0, 256, 0))
            x1,y1,x2,y2,x3,y3,x4,y4 = tuple([element for a in positions for element in a])
            canvas.rectangle([(x1 - 10, y1 - 10), (x1 + 10, y1 + 10)], width = 1, outline = (256, 0, 0))
            canvas.rectangle([(x2 - 10, y2 - 10), (x2 + 10, y2 + 10)], width = 1, outline = (256, 0, 0))
            canvas.rectangle([(x3 - 10, y3 - 10), (x3 + 10, y3 + 10)], width = 1, outline = (256, 0, 0))
            canvas.rectangle([(x4 - 10, y4 - 10), (x4 + 10, y4 + 10)], width = 1, outline = (256, 0, 0))
            if show:
                image.show()
            return image

def testBatchCutModel():
    image_objects = []
    for i in range(20):
        image_objects.append(testCutModel())
    fig , axs = plt.subplots(4,5,figsize = (12,4))
    for i,ax in enumerate(axs.flatten()):
        ax.imshow(image_objects[i])
    plt.show()
    plt.close()

def trainDetectModel(epoch=5):
    for i in range(epoch):
        epoch_loss = 0.0
        for index,(images,labels,_) in enumerate(myDetectDataLoader):
            outputs = net1(images)
            loss = lossModel1(outputs,labels)
            optimizerModel1.zero_grad()
            loss.backward()
            optimizerModel1.step()
            #print("epoch",i,"index",index,"loss",loss.item())
            epoch_loss += loss.item()
            
        print("epoch",i+1,"loss",epoch_loss / int(4000/batch_size_detect))
        print("-----------------------------")
        epoch_loss = 0.0
    torch.save(net1.state_dict(),'./model/model1.pth')
    print("Save PTH Model1 Success!")

def testDetectModel():
    net1.eval()
    for index,(images,_,name) in enumerate(myDetectTestDataLoader):
            output = net1(images)
            print(output)
            print(_)
            predicted = torch.argmax(output,dim=1) + 1
            image = Image.open("./data/train/{}".format(name[0])).convert("RGB")
            canvans = ImageDraw.Draw(image)
            canvans.text((14,14),str(int(predicted[0])),(0,256,0))
            return image
    
def saveAllModel():
    torch.save(net1,"./model/model1All.pth")
    torch.save(net2,"./model/model2All.pth")

if __name__ == "__main__":
    generateImages()
    


