import torch
import torchvision
from torch.autograd import Variable
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torchvision import transforms
import os
from model import ResNet101_net
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image


data_dir = 'F:/PycharmProject/pythonProject6/train'
csv_file = 'F:/PycharmProject/pythonProject6/trainLabels.csv'

# 自定义的数据加载器
class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_info = pd.read_csv(csv_file)  # 读取CSV文件
        self.class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data_info.iloc[idx, 0]) + '.png')  # 图像文件名
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[self.data_info.iloc[idx, 1]]  # 获取标签
        return image, label


# 创建图片变化形式
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(40, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomRotation(degrees=(-45, 45)),  # 在[-45, 45]度范围内随机旋转图像
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

data_transform2 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])


# 创建自定义数据集实例，将数据集加载进入
custom_dataset = CustomDataset(root_dir=data_dir, csv_file=csv_file, transform=data_transform)
custom_dataset2 = CustomDataset(root_dir=data_dir, csv_file=csv_file,transform=data_transform2)

# 将数据集拆分
total_size = len(custom_dataset)
train_size = int(0.99 * total_size)
valid_size = total_size - train_size
train_dataset, valid_dataset = random_split(custom_dataset2, [train_size, valid_size])



#创建数据加载器
train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

#定义模型，这里加载模型是多次训练，我先注释掉
model =  ResNet101_net()
# model.load_state_dict(torch.load('resnet101_model_final_dropout2_layer3.pth'))


# 将模型设置为训练模式并打印观察模型结构
model.train()
print(model)

# 将代码放入GPU上运行
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

# 使用交叉熵损失函数
loss_f = torch.nn.CrossEntropyLoss()

# 使用SGD优化器同时进行学习率梯度衰减
parm = model.parameters()
optimizer = SGD(parm,lr=0.0014, momentum=0.857142,
                                 weight_decay=0.000857142, nesterov=False)
scheduler = StepLR(optimizer, step_size=4, gamma=0.95142)#更新学习率


# Train函数进行训练
def train(epoch_n = 150):
    for epoch in range(epoch_n):
        print('epoch {}/{}'.format(epoch, epoch_n - 1))
        print('-' * 10)
        scheduler.step()  # 更新学习率

        # 获取当前学习率并且打印
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch}/{epoch_n - 1}], Current Learning Rate: {current_lr}')
        for phase in ['train', 'valid']:
            if phase == 'train':
                print('training...')
                model.train(True)
                data_loader = train_loader
            else:
                print('validing...')
                model.train(False)
                data_loader = valid_loader

            running_loss = 0.0
            running_corrects = 0.0

            for batch, data in enumerate(data_loader, 1):
                X, Y = data
                if use_gpu:
                    X, Y = Variable(X).cuda(), Variable(Y).cuda()
                else:
                    X, Y = Variable(X), Variable(Y)


                y_pred = model(X)
                _, pred = torch.max(y_pred.data, 1)
                optimizer.zero_grad()
                loss = loss_f(y_pred, Y)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data.item()
                running_corrects += torch.sum(pred == Y.data)
                if batch % 500 == 0 and phase == 'train':
                    current_batch_size = X.size(0)
                    print('batch {}, trainLoss: {:.4f}, trainAcc: {:.4f}'.format(
                        batch, running_loss / batch, 100 * running_corrects / (current_batch_size * batch)))

            epoch_loss = running_loss / len(data_loader)
            epoch_acc = 100 * running_corrects / len(data_loader.dataset)
            print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))
        torch.save(model,'result.pth')



# 主函数入口
if __name__ == '__main__':
    train()

