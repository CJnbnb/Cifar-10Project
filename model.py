from torchvision.models import resnet101
from torchvision.models.resnet import Bottleneck
import torch
import torch.nn as nn


class ResNet101_net(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(ResNet101_net, self).__init__()

        # 加载预训练的ResNet-101模型
        model = resnet101(pretrained=False)  # 不使用预训练权重

        # 修改第一个卷积层，使其接受3个通道输入
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 将最大池化层修改为恒等映射
        model.maxpool = nn.Identity()

        # 修改全连接层以输出10个类别
        model.fc = nn.Linear(model.fc.in_features, 10)

        # 在基本块中添加Dropout
        self.apply_dropout_to_blocks(model.layer1, dropout_prob)
        self.apply_dropout_to_blocks(model.layer2, dropout_prob)
        self.apply_dropout_to_blocks(model.layer3, dropout_prob)
        self.apply_dropout_to_blocks(model.layer4, dropout_prob)

        self.resnet101 = model

    def apply_dropout_to_blocks(self, blocks, dropout_prob):
        for block in blocks:
            if isinstance(block, Bottleneck):
                block.conv1 = nn.Sequential(
                    block.conv1,
                    nn.Dropout(p=dropout_prob)
                )
                block.conv2 = nn.Sequential(
                    block.conv2,
                    nn.Dropout(p=dropout_prob)
                )
                block.conv3 = nn.Sequential(
                    block.conv3,
                    nn.Dropout(p=dropout_prob)
                )

    def forward(self, x):
        x = self.resnet101(x)
        x = torch.flatten(x, 1)  # 将特征张量展平为一维张量
        return x