import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
from sklearn.cluster import KMeans
import math
from copy import deepcopy


# import brevitas.nn as qnn

bit = 4


def k_means(weight, clusters):
    temp_shape = weight.shape
    weight = weight.reshape(-1, 1)
    km = KMeans(n_clusters=clusters, n_init=1, max_iter=50)
    km.fit(weight)
    centers = km.cluster_centers_
    labels = km.labels_.reshape(temp_shape)
    return torch.from_numpy(centers).view(1, -1), torch.from_numpy(labels).int()


def transform(centers, labels):
    weight = torch.zeros_like(labels).float()
    for i, c in enumerate(centers.numpy().squeeze()):
        weight[labels == i] = c.item()
    return weight


class MyLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__(in_features, out_features)
        self.mask = None
        self.weight_labels = None
        self.elements = None

    def set_mask(self, mask):
        self.mask = mask
        # print(self.weight.data)
        # print(self.mask.data)
        self.weight.data = self.weight.data * self.mask.data

    def quant(self, bits=4):
        self.elements = 2 ** bits
        w = self.weight.data
        r = k_means(w.numpy(), self.elements)
        self.weight_labels = r[1]
        self.weight.data = transform(r[0], r[1]).float()

    def forward(self, input: Tensor) -> Tensor:
        if self.mask is not None:
            return F.linear(input, self.weight * self.mask, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)


class MyConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MyConv2d, self).__init__(in_channels, out_channels,
                                       kernel_size, stride, padding, dilation, groups, bias)
        self.weight_labels = None
        self.num_cent = None

    def quant(self, bits=4):
        self.elements = 2 ** bits
        w = self.weight.data
        r = k_means(w.numpy(), self.elements)
        self.weight_labels = r[1]
        self.weight.data = transform(r[0], r[1]).float()


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = MyConv2d(1, 6, 5)
        self.conv2 = MyConv2d(6, 16, 5)
        # self.conv3 = MyConv2d(64, 64, kernel_size=3, padding=1, stride=1)

        self.fc1 = MyLinear(256, 147)
        # self.fc2 = MyLinear(120, 84)
        self.fc3 = MyLinear(147, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = F.relu(self.conv3(x))
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def kmeans_quant(self, bits=4):
        self.conv1.quant(bits)
        self.conv2.quant(bits)
        # self.conv3.quant(bits)
        self.fc1.quant(bits)
        self.fc3.quant(bits)

    def set_masks(self, masks):
        self.fc1.set_mask(masks[2])
        # self.fc2.set_mask(masks[1])
        self.fc3.set_mask(masks[3])


# n = 7  # fractional part


# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.conv1 = qnn.QuantConv2d(1, 6, 5,
#                                      bias=False,
#                                      weight_quant_type=QuantType.INT,
#                                      weight_bit_width=bit,
#                                      weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
#                                      weight_scaling_impl_type=ScalingImplType.CONST,
#                                      weight_scaling_const=1.0)
#         self.relu1 = qnn.QuantReLU(quant_type=QuantType.INT,
#                                    bit_width=bit,
#                                    max_val=1 - 1 / 128.0,
#                                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
#                                    scaling_impl_type=ScalingImplType.CONST)

#         self.conv2 = qnn.QuantConv2d(6, 16, 5,
#                                      bias=False,
#                                      weight_quant_type=QuantType.INT,
#                                      weight_bit_width=bit,
#                                      weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
#                                      weight_scaling_impl_type=ScalingImplType.CONST,
#                                      weight_scaling_const=1.0)

#         self.relu2 = qnn.QuantReLU(quant_type=QuantType.INT,
#                                    bit_width=bit,
#                                    max_val=1 - 1 / 128.0,
#                                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
#                                    scaling_impl_type=ScalingImplType.CONST)
#         self.fc1 = qnn.QuantLinear(256, 147,
#                                    bias=True,
#                                    weight_quant_type=QuantType.INT,
#                                    weight_bit_width=bit,
#                                    weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
#                                    weight_scaling_impl_type=ScalingImplType.CONST,
#                                    weight_scaling_const=1.0)
#         self.relu3 = qnn.QuantReLU(quant_type=QuantType.INT,
#                                    bit_width=bit,
#                                    max_val=1 - 1 / 128.0,
#                                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
#                                    scaling_impl_type=ScalingImplType.CONST)
#         self.fc3 = qnn.QuantLinear(147, 10,
#                                    bias=True,
#                                    weight_quant_type=QuantType.INT,
#                                    weight_bit_width=bit,
#                                    weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
#                                    weight_scaling_impl_type=ScalingImplType.CONST,
#                                    weight_scaling_const=1.0)

#     def forward(self, x):
#         x = F.max_pool2d(self.relu1(self.conv1(x)), 2)
#         x = F.max_pool2d(self.relu2(self.conv2(x)), 2)
#         # x = F.relu(self.conv3(x))
#         x = x.view(-1, int(x.nelement() / x.shape[0]))
#         x = self.relu3(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, 256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm1d(256),

        )

        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.softmax(self.fc2((self.fc(x))), dim=1)
        return x
    #     self.conv1 = nn.Conv2d(1, 16, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(16, 16, 5)
    #     self.fc1 = nn.Linear(16 * 9 * 9, 256)
    #     self.fc2 = nn.Linear(256, 256)
    #     self.fc3 = nn.Linear(256, 10)
    #
    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     # x = x.view(-1, 16 * 9 * 9)
    #     x = x.view(-1, int(x.nelement() / x.shape[0]))
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

        # 1 input image channel, 6 output channels, 3x3 square conv kernel
    #     self.conv1 = nn.Conv2d(1, 32, 3, 1)
    #     self.conv2 = nn.Conv2d(32, 64, 3, 1)
    #     self.conv3 = nn.Conv2d(64, 64, 3, 1)
    #     self.dropout1 = nn.Dropout2d(0.3)
    #     self.dropout2 = nn.Dropout2d(0.5)
    #     self.fc1 = nn.Linear(9216, 128)
    #     self.fc2 = nn.Linear(128, 10)
    #
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = F.relu(x)
    #     x = self.conv2(x)
    #     x = F.relu(x)
    #     x = F.max_pool2d(x, 2)
    #     x = self.dropout1(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.dropout2(x)
    #     output = self.fc2(x)
    #     return output
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.conv3 = nn.Conv2d(16, 120, 4)
    #     self.dropout1 = nn.Dropout2d(0.3)
    #     self.dropout2 = nn.Dropout2d(0.5)
    #     self.fc1 = nn.Linear(120, 1024)  # 5x5 image dimension
    #     self.fc2 = nn.Linear(1024, 256)
    #     self.fc3 = nn.Linear(256, 10)
    #
    # def forward(self, x):
    #     x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=2)
    #     x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    #     x = self.conv3(x)
    #     x = self.dropout1(x)
    #     # x = F.max_pool2d(F.relu(self.conv3(x)), 2)
    #     x = x.view(-1, int(x.nelement() / x.shape[0]))
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.dropout2(x)
    #     x = self.fc3(x)
    #     return x
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.conv2 = nn.Conv2d(6, 6, 5)
    #     self.fc3 = nn.Linear(96, 10)
    #
    # def forward(self, x):
    #     x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    #     x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    #     x = x.view(-1, int(x.nelement() / x.shape[0]))
    #     # x = F.relu(self.fc1(x))
    #     # x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x


class YourNet(nn.Module):
    def __init__(self):
        super(YourNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.fc3 = nn.Linear(96, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = self.fc3(x)
        return x
