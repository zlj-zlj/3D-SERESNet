import numpy as np
import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F


class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, (1, kernel_size, kernel_size), (1, stride, stride),
                              (0, padding, padding), bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.initialize_weights()

    def initialize_weights(self):

        init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, num_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, num_channels, 1, stride=1,padding=0)
        self.conv2 = ConvLayer(num_channels, num_channels, 3, stride=stride, padding=1)
        self.conv3 = ConvLayer(num_channels, out_channels, 1, stride=1,padding=0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = ConvLayer(in_channels, out_channels, 1, stride=stride)

        self.initialize_weights()

    def initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        # print('1: ',x.shape)
        identity = x
        x = self.conv1(x)
        # x = self.relu(x)
        x= F.leaky_relu(x)
        x = self.conv2(x)
        # x = self.relu(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        # print('2: ',x.shape)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        # print('3: ', x.shape)
        # x = self.relu(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return x

class SEFRBNet(nn.Module):

    def __init__(self):
        super(SEFRBNet1201, self).__init__()
        self.conv1 = ConvLayer(1, 16, 7, stride=2, padding=3)
        self.pool1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ResBlock(16, 16, 32),
            SEAttention(32)
        )
        self.conv2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.layer2 = nn.Sequential(
            ResBlock(32, 32, 64, stride=2),
            SEAttention(64)
        )
        self.conv3 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.layer3 = nn.Sequential(
            ResBlock(64, 64, 128, stride=2),
            SEAttention(128)
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv5 = nn.Conv3d(128, 32, 1)
        self.bn5 = nn.BatchNorm3d(32)
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 2)

    def forward(self, x):  # x=(80,1,16,45,79)
        x = self.conv1(x)  # x=21 1 16 129 79->24 16 16 65 40
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])  # x=24 256 65 40
        x = self.pool1(x)  # x=24 256 33 20
        x = x.reshape(x.shape[0], 16, -1, x.shape[2], x.shape[3])  # x=24 16 16 33 20

        x = self.layer1(x)  # x=24 32 16 33 20
        x = self.conv2(x)  # x=24 32 16 33 20
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer2(x)  # x=24 64 16 17 10
        x = self.conv3(x)  # x=24 64 16 17 10
        x = self.bn3(x)
        x = F.relu(x)
        x = self.layer3(x)  # x=24 128 16 9 5
        x = self.conv5(x)  # x=24 32 16 9 5
        x = self.bn5(x)
        x = F.relu(x)
        x = self.avgpool(x)  # x=24 32 1 1 1
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = F.sigmoid(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear2(x)
        return x  # 鏉╂柨娲栨潏鎾冲毉






