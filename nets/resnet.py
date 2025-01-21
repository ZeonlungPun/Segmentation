import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        #-----------------------------------------------------------#
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        #-----------------------------------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 600,600,3 -> 300,300,64
        self.conv1  = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 38,38,1024 -> 19,19,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x       = self.conv1(x)
        x       = self.bn1(x)
        feat1   = self.relu(x)

        x       = self.maxpool(feat1)
        feat2   = self.layer1(x)

        feat3   = self.layer2(feat2)
        feat4   = self.layer3(feat3)
        feat5   = self.layer4(feat4)
        # 2x,2x,4x,16x,
        # print(feat1.shape)
        # print(feat2.shape)
        # print(feat3.shape)
        # print(feat4.shape)
        # print(feat5.shape)
        return [feat1, feat2, feat3, feat4, feat5]

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'), strict=False)
    
    del model.avgpool
    del model.fc
    return model


import torchvision.models as models
class ResNet18Custom(nn.Module):
    def __init__(self):
        super(ResNet18Custom, self).__init__()
        # Load the pre-trained ResNet18 model without the final fully connected layer
        self.resnet18 = models.resnet18(pretrained=False)

        # Remove the fully connected layer (fc)
        self.resnet18 = nn.Sequential(
            *list(self.resnet18.children())[:-1]  # Remove the last fc layer
        )
        self.tran1=nn.Conv2d(128,256,kernel_size=1)
        self.tran2=nn.Conv2d(256,512,kernel_size=1)
        self.tran3 = nn.Conv2d(512, 1024, kernel_size=1)
        self.layer4=nn.Conv2d(512,2048,kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        # Pass through the layers to extract feature maps
        x = self.resnet18[0](x)  # conv1
        feat1 = self.resnet18[1](x)  # bn1 + relu
        x = self.resnet18[4](feat1)  # maxpool
        feat2 = self.resnet18[5](x)  # layer1
        feat2_=self.tran1(feat2)
        feat3 = self.resnet18[6](feat2)  # layer2
        feat3_=self.tran2(feat3)
        feat4 = self.resnet18[7](feat3)  # layer3
        feat4_=self.tran3(feat4)
        feat5 = self.layer4(feat4)
        # print(feat1.shape)
        # print(feat2_.shape)
        # print(feat3_.shape)
        # print(feat4_.shape)
        # print(feat5.shape)
        # Return feature maps at different stages (before fc)
        return [feat1, feat2_, feat3_, feat4_, feat5]
def resnet18():
    model=ResNet18Custom()
    return model


class ResNet34Custom(nn.Module):
    def __init__(self):
        super(ResNet34Custom, self).__init__()
        # Load the pre-trained ResNet34 model without the final fully connected layer
        self.resnet34 = models.resnet34(pretrained=True)

        # Remove the fully connected layer (fc)
        self.resnet34 = nn.Sequential(
            *list(self.resnet34.children())[:-1]  # Remove the last fc layer
        )
        self.tran1 = nn.Conv2d(128, 256, kernel_size=1)
        self.tran2 = nn.Conv2d(256, 512, kernel_size=1)
        self.tran3 = nn.Conv2d(512, 1024, kernel_size=1)
        self.layer4 = nn.Conv2d(512, 2048, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        # Pass through the layers to extract feature maps
        x = self.resnet34[0](x)  # conv1
        feat1 = self.resnet34[1](x)  # bn1 + relu
        x = self.resnet34[4](feat1)  # maxpool
        feat2 = self.resnet34[5](x)  # layer1
        feat2_ = self.tran1(feat2)
        feat3 = self.resnet34[6](feat2)  # layer2
        feat3_ = self.tran2(feat3)
        feat4 = self.resnet34[7](feat3)  # layer3
        feat4_ = self.tran3(feat4)
        feat5 = self.layer4(feat4)
        # print(feat1.shape)
        # print(feat2_.shape)
        # print(feat3_.shape)
        # print(feat4_.shape)
        # print(feat5.shape)
        # Return feature maps at different stages (before fc)
        return [feat1, feat2_, feat3_, feat4_, feat5]
def resnet34():
    model=ResNet34Custom()
    return model


class ResNet101Custom(nn.Module):
    def __init__(self):
        super(ResNet101Custom, self).__init__()
        # Load the pre-trained ResNet101 model without the final fully connected layer
        self.resnet101 = models.resnet101(pretrained=True)

        # Remove the fully connected layer (fc)
        self.resnet101 = nn.Sequential(
            *list(self.resnet101.children())[:-1]  # Remove the last fc layer
        )

    def forward(self, x):
        # Pass through the layers to extract feature maps
        x = self.resnet101[0](x)  # conv1
        feat1 = self.resnet101[1](x)  # bn1 + relu
        x = self.resnet101[3](feat1)  # maxpool
        feat2 = self.resnet101[4](x)  # layer1
        feat3 = self.resnet101[5](feat2)  # layer2
        feat4 = self.resnet101[6](feat3)  # layer3
        feat5 = self.resnet101[7](feat4)  # layer4
        # print(feat1.shape)
        # print(feat2.shape)
        # print(feat3.shape)
        # print(feat4.shape)
        # print(feat5.shape)
        # Return feature maps at different stages (before fc)
        return [feat1, feat2, feat3, feat4, feat5]
def resnet101():
    model=ResNet101Custom()
    return model
if __name__ =='__main__':
    import torch
    x=torch.rand(1,3,600,600)
    resnet34().forward(x)