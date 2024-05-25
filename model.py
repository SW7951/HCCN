'''SWMobileNet in PyTorch.'''
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torch.distributions.multivariate_normal import MultivariateNormal

from utils.io_util import read_object


class hswish(nn.Module):
    """ hard swish激活函数"""
    def forward(self, x):
        """ """
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    """ hard sigmod激活函数"""
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    """ SE注意力模块 """
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 cx1x1
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        # 相当于对每个channel乘一个权重
        return x * self.se(x)  # cxhxw, cx1x1


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000, act=nn.Hardswish):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 2),
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(576, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.linear4 = nn.Linear(1280, num_classes)
        self.sw = SwLayer(num_classes)  # 相似性权重层
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        prob = self.sw(x)  # 计算相似度权重

        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten(1)
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))

        out = self.linear4(out) + prob
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000, act=nn.Hardswish, alpha=0.73):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)
        self.alpha = alpha

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, False, 1), #
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(3, 40, 240, 80, act, False, 2),
            Block(3, 80, 200, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 480, 112, act, True, 1),
            Block(3, 112, 672, 112, act, True, 1),
            Block(5, 112, 672, 160, act, True, 2),
            Block(5, 160, 672, 160, act, True, 1),
            Block(5, 160, 960, 160, act, True, 1),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(960, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)

        self.linear4 = nn.Linear(1280, num_classes)
        self.sw = SwLayer(num_classes)  # 相似性权重层
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.hs1(self.bn1(self.conv1(x)))
        out1 = self.bneck(out1)

        out1 = self.hs2(self.bn2(self.conv2(out1)))
        out1 = self.gap(out1).flatten(1)
        out1 = self.drop(self.hs3(self.bn3(self.linear3(out1))))

        out1 = self.linear4(out1)  # 跳连线
        out2 = self.sw(x)  # 跳连线

        return self.alpha * out1 + (1 - self.alpha) * out2

class SwLayer(nn.Module):
    ''' 相似性权重层 Similarity Weights Layer '''
    # Class Affinity Weights
    def __init__(self, num_classes: int):
        super(SwLayer, self).__init__()
        self.num_classes = num_classes
        # 定义参数
        # self.k = nn.Parameter(torch.ones(num_classes, dtype=torch.float32))
        self.fc1 = nn.Linear(num_classes, 512)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 512)
        self.act2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(512, num_classes)
        # 注册缓冲区
        ms: torch.Tensor = read_object('./data/mean1.pkl')
        cs: torch.Tensor = read_object('./data/conv1.pkl')
        assert len(ms) == len(cs)
        self.register_buffer(f'means', ms)
        self.register_buffer(f'convs', cs)

    def forward(self, x):
        # 以均值为指标
        value = torch.mean(x, dim=(2, 3), dtype=torch.float)  # [N, C, H, W] -> [N, C]
        probs = torch.stack(
            [MultivariateNormal(self.means[i], covariance_matrix=self.convs[i]).log_prob(value).exp() \
             for i in range(self.num_classes)],
            dim=1
        )
        # out = self.act1(self.fc1(probs))
        out = self.drop1(self.act1(self.fc1(probs)))
        # out = self.act2(self.fc2(out))
        out = self.drop2(self.act2(self.fc2(out)))
        out = self.fc3(out)
        return out


def get_model_finetuning_the_convnet(num_classes, alpha=0.73):
    """
    获取卷积层微调的mobile netv3模型
    :param num_classes:
    :return:
    """
    model = MobileNetV3_Large(alpha=alpha)
    model.load_state_dict(torch.load("data/450_act3_mobilenetv3_large.pth"), strict=False)  # 下载预训练权重

    in_channels = model.linear4.in_features  # 获得最后fc层的in_features参数
    model.linear4 = nn.Linear(in_channels, num_classes)  # 改变原网络最后一层参数
    model.sw = SwLayer(num_classes)
    return model


if __name__ == '__main__':
    model = get_model_finetuning_the_convnet(7).to('cuda')
    print(model)
    x = torch.rand((32, 3, 224, 224)).to('cuda')
    pred = model(x)
    print(pred.shape)