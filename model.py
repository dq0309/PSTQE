import math
import torch.nn.functional as F

from torch import nn
import torch


class STNet(nn.Module):
    def __init__(self, channels):
        super(STNet, self).__init__()

        self.Layer_C3D_1 = C3D_Layer(channels)

        self.Module_3D_1 = CNN3D_Module(channels)

        self.Module_3D_2 = CNN3D_Module(channels)

        self.Module_3D_3 = CNN3D_Module(channels)

        self.Temporal_senet1 = Sigmoid_SELayer_Temporal(temporal_channel=5, reduction=1)

        self.Spatial_att1 = Sigmoid_Spatial_Temporal(temporal_channel=5, increase=2)

        self.Module_3D_4 = CNN3D_Module(channels)

        self.t_pool = nn.MaxPool3d(kernel_size=(5, 1, 1), stride=(1, 1, 1))

        # self.t_conv1 = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1,
        #                        bias=True)

        self.s_conv1 = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.s_prelu1 = nn.PReLU(channels)

        self.s_block1 = RecursiveRDB(channels)

        self.s_conv3 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0,
                                 bias=True)

        self.s_prelu3 = nn.PReLU(channels)

        self.s_block2 = RecursiveRDB(channels)

        self.s_conv4 = nn.Conv2d(in_channels=channels * 3, out_channels=channels, kernel_size=1, stride=1, padding=0,
                                 bias=True)

        self.s_prelu4 = nn.PReLU(channels)

        self.st_conv1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0,
                                  bias=True)

        self.st_prelu1 = nn.PReLU(channels)

        self.st_CA = Sigmoid_SELayer(channel=channels)

        self.st_SA = Sigmoid_Spatial()

        self.st_dense = DenseBlock(channels=channels)

        self.st_conv2 = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1,
                                  bias=True)

    def forward(self, im0, im1, im2, im3, im4):
        s_feature1 = self.s_conv1(im2)

        s_feature1 = self.s_prelu1(s_feature1)

        # s_feature1 = self.s_senet1(s_feature1)

        s_feature2 = self.s_block1(s_feature1)

        s_feature3 = torch.cat([s_feature1, s_feature2], 1)

        # s_feature3 = self.s_senet2(s_feature3)

        s_feature3 = self.s_conv3(s_feature3)

        s_feature3 = self.s_prelu3(s_feature3)

        s_feature3 = self.s_block2(s_feature3)

        s_feature4 = torch.cat([s_feature1, s_feature2, s_feature3], 1)

        # s_feature4 = self.s_senet3(s_feature4)

        s_feature4 = self.s_conv4(s_feature4)

        s_feature4 = self.s_prelu4(s_feature4)

        im0 = torch.unsqueeze(im0, 1)
        im1 = torch.unsqueeze(im1, 1)
        im2 = torch.unsqueeze(im2, 1)
        im3 = torch.unsqueeze(im3, 1)
        im4 = torch.unsqueeze(im4, 1)

        video = torch.cat([im0, im1, im2, im3, im4], 2)

        video = self.Layer_C3D_1(video)

        video_1 = self.Module_3D_1(video)

        video_1 = self.Module_3D_2(video_1)

        video_1 = self.Module_3D_3(video_1)

        video_1 = self.Module_3D_4(video_1)

        video_1 = self.Temporal_senet1(video_1)

        video_1 = self.Spatial_att1(video_1)

        # video_1 = video_1 * video * self.gamma + video

        video_1 = video_1 * video

        video_1 = self.t_pool(video_1)

        video_1 = torch.squeeze(video_1, 2)

        video_1 = torch.cat([video_1, s_feature4], 1)

        video_1 = self.st_conv1(video_1)

        video_1 = self.st_prelu1(video_1)

        y = self.st_dense(video_1)

        y = self.st_CA(y)

        y = self.st_SA(y)

        video_1 = video_1 * y

        video_1 = self.st_conv2(video_1)

        im2 = torch.squeeze(im2, 2)

        return im2 + video_1


class CNN3D_Module(nn.Module):
    def __init__(self, channels):
        super(CNN3D_Module, self).__init__()

        self.net_3d = nn.Sequential(

            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True),

            nn.PReLU(channels),

        )

    def forward(self, x):
        return self.net_3d(x)


class C3D_Layer(nn.Module):
    def __init__(self, channels):
        super(C3D_Layer, self).__init__()

        self.net_C3D = nn.Sequential(

            nn.Conv3d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True),

            nn.PReLU(channels),

        )

    def forward(self, x):
        return self.net_C3D(x)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.prelu1 = nn.PReLU(channels)

        self.conv2 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               bias=True)

        self.prelu2 = nn.PReLU(channels)

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.prelu3 = nn.PReLU(channels)

        self.conv4 = nn.Conv2d(in_channels=channels * 3, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               bias=True)

        self.prelu4 = nn.PReLU(channels)

        self.conv5 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.prelu5 = nn.PReLU(channels)

        self.conv6 = nn.Conv2d(in_channels=channels * 4, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               bias=True)

        self.prelu6 = nn.PReLU(channels)

    def forward(self, x):
        feature1 = self.conv1(x)

        feature1 = self.prelu1(feature1)

        feature2 = torch.cat([x, feature1], 1)

        feature2 = self.conv2(feature2)

        feature2 = self.prelu2(feature2)

        feature2 = self.conv3(feature2)

        feature2 = self.prelu3(feature2)

        feature3 = torch.cat([x, feature1, feature2], 1)

        feature3 = self.conv4(feature3)

        feature3 = self.prelu4(feature3)

        feature3 = self.conv5(feature3)

        feature3 = self.prelu5(feature3)

        feature3 = torch.cat([x, feature1, feature2, feature3], 1)

        feature3 = self.conv6(feature3)

        feature3 = self.prelu6(feature3)

        return feature3


class Sigmoid_SELayer_Temporal(nn.Module):
    def __init__(self, temporal_channel, reduction=1):
        super(Sigmoid_SELayer_Temporal, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(

            nn.Linear(temporal_channel, temporal_channel // reduction, bias=False),

            nn.ReLU(),

            nn.Linear(temporal_channel // reduction, temporal_channel, bias=False),

            nn.Sigmoid()

        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = x.permute(0, 2, 1, 3, 4)

        b, t, c, h, w = y.size()

        y = self.avg_pool(y)

        y = y.view(b, t)

        y = self.fc(y)

        y = y.unsqueeze(2)
        y = y.unsqueeze(2)
        y = y.unsqueeze(2)

        y = y.permute(0, 2, 1, 3, 4)

        y = y.expand_as(x)

        return x * y * self.gamma


class Sigmoid_Spatial_Temporal(nn.Module):
    def __init__(self, temporal_channel, increase=2):
        super(Sigmoid_Spatial_Temporal, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(temporal_channel, 1, 1), stride=1,
                               padding=0, bias=True)

        self.relu1 = nn.ReLU(8)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1,
                               padding=1, bias=True)

        self.relu2 = nn.ReLU(8)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1,
                               padding=1, bias=True)

        self.relu3 = nn.ReLU(8)

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1,
                               padding=0, bias=True)

        self.sigmoid = nn.Sigmoid()

        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = self.conv1(x)

        y = self.relu1(y)

        y = y.squeeze(2)

        y = self.conv2(y)

        y = self.relu2(y)

        y = self.conv3(y)

        y = self.relu3(y)

        y = self.conv4(y)

        y = self.sigmoid(y)

        y = y.unsqueeze(2)

        y = y.expand_as(x)

        # y = x * y * self.gamma + x

        return y


class RecursiveBlock(nn.Module):
    def __init__(self, channels):
        super(RecursiveBlock, self).__init__()

        self.net = nn.Sequential(

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),

            nn.PReLU(channels),

            nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2, ),

            nn.PReLU(channels),

            # nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3),
            #
            # nn.PReLU(channels),

        )

    def forward(self, x):
        return self.net(x)


class RecursiveRDB(nn.Module):
    def __init__(self, channels):
        super(RecursiveRDB, self).__init__()

        self.recursiveblock = RecursiveBlock(channels)

        self.conv1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               bias=True)

        self.prelu1 = nn.PReLU(channels)

        self.conv2 = nn.Conv2d(in_channels=channels * 3, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               bias=True)

        self.prelu2 = nn.PReLU(channels)

        self.conv3 = nn.Conv2d(in_channels=channels * 4, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               bias=True)

        self.prelu3 = nn.PReLU(channels)

    def forward(self, x):
        im_1 = self.recursiveblock(x)

        im_2 = torch.cat([x, im_1], 1)

        im_2 = self.conv1(im_2)

        im_2 = self.prelu1(im_2)

        im_2 = self.recursiveblock(im_2)

        im_3 = torch.cat([x, im_1, im_2], 1)

        im_3 = self.conv2(im_3)

        im_3 = self.prelu2(im_3)

        im_3 = self.recursiveblock(im_3)

        im_4 = torch.cat([x, im_1, im_2, im_3], 1)

        im_4 = self.conv3(im_4)

        im_4 = self.prelu3(im_4)

        return im_4 + x


class Sigmoid_SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Sigmoid_SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channel, channel // reduction, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(channel // reduction, channel, bias=False),

            nn.Sigmoid()

        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y)


        y = y.view(b, c, 1, 1)

        y = y.expand_as(x)

        return x * y * self.gamma


class Sigmoid_Spatial(nn.Module):
    def __init__(self):
        super(Sigmoid_Spatial, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1,1), stride=1, padding=0, bias=True)

        self.relu1 = nn.ReLU(8)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1,
                               padding=1, bias=True)

        self.relu2 = nn.ReLU(8)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1,
                               padding=1, bias=True)

        self.relu3 = nn.ReLU(8)

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1,
                               padding=0, bias=True)

        self.sigmoid = nn.Sigmoid()

        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = self.conv1(x)

        y = self.relu1(y)

        # y = y.squeeze(2)

        y = self.conv2(y)

        y = self.relu2(y)

        y = self.conv3(y)

        y = self.relu3(y)

        y = self.conv4(y)

        y = self.sigmoid(y)

        # y = y.unsqueeze(2)

        y = y.expand_as(x)

        # y = x * y * self.gamma + x

        return y