import csv
import os

import torch
import torchvision
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import ResNet101_net

cifar10_labels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}



# 定义数据转换
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    ])


class CustomDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.filenames = os.listdir(folder)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.folder, filename)
        image = Image.open(filepath)
        if self.transform:
            image = self.transform(image)
        return image, filename

    # 创建Dataset和DataLoader


test_folder = 'F:/PycharmProject/fishdetect2/test'
test_dataset = CustomDataset(test_folder, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)



model = ResNet101_net()
# 模型设置为评估模式

model.load_state_dict(torch.load('resnet101_model_final_dropout2_layer3.pth'))
model.eval()


use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

predictions = []
with torch.no_grad():
    for images, filenames in test_loader:
        if use_gpu:
            images = images.cuda()
        y_pred = model(images)
        _, preds = torch.max(y_pred, 1)
        for filename, pred in zip(filenames, preds):
            idx = os.path.splitext(filename)[0]
            predictions.append((idx, pred.item()))

# 写入CSV文件
csv_filename = 'result.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])
    for idx, label_pred in predictions:
        label = cifar10_labels[label_pred]
        writer.writerow([idx, label])

print(f"预测结果已保存到 {csv_filename} 文件中。")