import torch
import torch.nn as nn
import torch.nn.functional as F

from block.CBAM import CBAM
from block.blocks import StripCGLU
from block.ODconv import ODConv2d
from block.backbone.convnextv2 import convnextv2_tiny

from block.backbone.repvit import repvit_m0_9

from block.backbone.starnet import starnet_s050

from block.conv import PConv, RFAConv
from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

#暂时无法使用unetConv
class unetConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetConv, self).__init__()

        self.conv1 = ODConv2d(in_size, out_size,kernel_size=3,padding=1)

        self.conv2 = ODConv2d(out_size, out_size,kernel_size=3,padding=1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        #outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        #outputs = self.relu(outputs)
        return outputs

class unetUpAttention(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUpAttention, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)

        self.attention = CBAM(c1=out_size)  ##将注意力机制添加到self.conv_cat之前

        #self.attention = StripCGLU(out_size)

        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.attention(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        elif backbone == "repvit_m0_9":
            self.repvit_m0_9 = repvit_m0_9()
            in_filters  = [176, 352, 704, 768]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50, convnextv2_tiny, efficientformerv2_s0, efficientvit_m0, fasternet_t0, lsknet_s, lsnet_t, mobilenetv4convsmall, overlock_t, pkinet_t, repvit_m0_9, rmt_t, starnet_s050, swintransformer_tiny, unireplknet_t.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        elif backbone == 'repvit_m0_9':
            self.up_conv = None
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        elif self.backbone == "repvit_m0_9":
            [feat1, feat2, feat3, feat4, feat5] = self.repvit_m0_9.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False
        elif self.backbone == "repvit_m0_9":
            for param in self.repvit_m0_9.parameters():
                param.requires_grad = False


    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
        elif self.backbone == "repvit_m0_9":
            for param in self.repvit_m0_9.parameters():
                param.requires_grad = True

class UnetAttention1(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(UnetAttention1, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        elif backbone == "repvit_m0_9":
            self.repvit_m0_9 = repvit_m0_9()
            in_filters = [176, 352, 704, 768]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50, convnextv2_tiny, efficientformerv2_s0, efficientvit_m0, fasternet_t0, lsknet_s, lsnet_t, mobilenetv4convsmall, overlock_t, pkinet_t, repvit_m0_9, rmt_t, starnet_s050, swintransformer_tiny, unireplknet_t.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        #self.up_concat4 = unetUpAttention(in_filters[3], out_filters[3])
        self.up_concat4 = unetConv(in_filters[3], out_filters[3])
        # 128,128,256
        #self.up_concat3 = unetUpAttention(in_filters[2], out_filters[2])
        self.up_concat3 = unetConv(in_filters[2], out_filters[2])
        # 256,256,128
        #self.up_concat2 = unetUpAttention(in_filters[1], out_filters[1])
        self.up_concat2 = unetConv(in_filters[1], out_filters[1])
        # 512,512,64
        #self.up_concat1 = unetUpAttention(in_filters[0], out_filters[0])
        self.up_concat1 = unetConv(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        elif backbone == 'repvit_m0_9':
            self.up_conv = None
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        elif self.backbone == "repvit_m0_9":
            [feat1, feat2, feat3, feat4, feat5] = self.repvit_m0_9.forward(inputs)


        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False
        elif self.backbone == "repvit_m0_9":
            for param in self.repvit_m0_9.parameters():
                param.requires_grad = False
    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True

        elif self.backbone == "repvit_m0_9":
            for param in self.repvit_m0_9.parameters():
                param.requires_grad = True





####原始的unet
class UNet_origin(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, c_attention=False, s_attention=False):
        super(UNet_origin, self).__init__()
        if c_attention:
            if s_attention:
                self.model_name = 'unet_cs'
            else:
                self.model_name = 'unet_c'
        elif s_attention:
            self.model_name = 'unet_s'
        else:
            self.model_name = 'unet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = c_attention or s_attention

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        if self.attention:
            self.cbam1 = CBAM(64, c_attention, s_attention)
            self.cbam2 = CBAM(128, c_attention, s_attention)
            self.cbam3 = CBAM(256, c_attention, s_attention)
            self.cbam4 = CBAM(512, c_attention, s_attention)

    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.cbam1(x1) + x1

        x2 = self.down1(x1)
        if self.attention:
            x2 = self.cbam2(x2) + x2

        x3 = self.down2(x2)
        if self.attention:
            x3 = self.cbam3(x3) + x3

        x4 = self.down3(x3)
        if self.attention:
            x4 = self.cbam4(x4) + x4

        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


#在连接处添加注意力机制
class UNet_origin_Attention1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNet_origin_Attention1, self).__init__()

        self.model_name = 'UNetAttention1'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        if self.attention:
            self.attention1 = CBAM(64)
            self.attention2 = CBAM(128)
            self.attention3 = CBAM(256)
            self.attention4 = CBAM(512)




            # self.attention1 = StripCGLU(64)
            # self.attention2 = StripCGLU(128)
            # self.attention3 = StripCGLU(256)
            # self.attention4 = StripCGLU(512)


    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.attention1(x1) + x1

        x2 = self.down1(x1)
        if self.attention:
            x2 = self.attention2(x2) + x2

        x3 = self.down2(x2)
        if self.attention:
            x3 = self.attention3(x3) + x3

        x4 = self.down3(x3)
        if self.attention:
            x4 = self.attention4(x4) + x4

        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits





#在卷积层中添加注意力机制
class UNet_origin_Attention2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_origin_Attention2, self).__init__()
        self.model_name = 'UNetAttention2'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConvAttention(n_channels, 64))
        self.down1 = (DownAttention(64, 128))   ##Down1中改变下采样的方式
        self.down2 = (DownAttention(128, 256))  ##Down是原始结构  DoubleConvAttention是添加注意力机制之后的结构 根据自己的需求选择
        self.down3 = (DownAttention(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

########新增
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #self.act = nn.SiLU()
        #self.act = nn.LeakyReLU(0.1)
        self.act = nn.ReLU(inplace=True)
        #self.act = MetaAconC(c2)
        #self.act = AconC(c2)
        #self.act = Mish()
        #self.act = Hardswish()
        #self.act = FReLU(c2)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))




class DoubleConvAttention(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)



        self.attention = CBAM(mid_channels)



        #self.attention = StripBlock(mid_channels)
        #self.attention = StripCGLU(mid_channels)


        self.conv2 = Conv(mid_channels, out_channels)



    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x

class DownAttention(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttention(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# class Down1(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             ADown(in_channels,in_channels),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bilinear=False):
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # TODO: bilinear
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(mid_channels, out_channels)

    def pad_and_concat(self, x1, *x2):
        # x1 is smaller, x2 is larger
        if len(x2) == 1:
            # input is CHW
            diffY = x2[0].size()[2] - x1.size()[2]
            diffX = x2[0].size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            return torch.cat([x2[0], x1], dim=1)
        else:
            # recursively pad and concat
            return self.pad_and_concat(x1, self.pad_and_concat(*x2))

    def forward(self, x1, *x2):
        x1 = self.up(x1)
        x = self.pad_and_concat(x1, *x2)
        return self.conv(x)


# class UpSample(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, mid_channels=None, bilinear=False):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = in_channels

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             # TODO: bilinear
#             #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.up = ADown(in_channels,in_channels // 2)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             #self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.up = ADown(in_channels,in_channels // 2)
#             self.conv = DoubleConv(mid_channels, out_channels)

#     def pad_and_concat(self, x1, *x2):
#         # x1 is smaller, x2 is larger
#         if len(x2) == 1:
#             # input is CHW
#             diffY = x2[0].size()[2] - x1.size()[2]
#             diffX = x2[0].size()[3] - x1.size()[3]
#             x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                             diffY // 2, diffY - diffY // 2])
#             return torch.cat([x2[0], x1], dim=1)
#         else:
#             # recursively pad and concat
#             return self.pad_and_concat(x1, self.pad_and_concat(*x2))

#     def forward(self, x1, *x2):
#         x1 = self.up(x1)
#         x = self.pad_and_concat(x1, *x2)
#         return self.conv(x)



class UpAttention(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bilinear=False):
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # TODO: bilinear
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvAttention(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvAttention(mid_channels, out_channels)

    def pad_and_concat(self, x1, *x2):
        # x1 is smaller, x2 is larger
        if len(x2) == 1:
            # input is CHW
            diffY = x2[0].size()[2] - x1.size()[2]
            diffX = x2[0].size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            return torch.cat([x2[0], x1], dim=1)
        else:
            # recursively pad and concat
            return self.pad_and_concat(x1, self.pad_and_concat(*x2))

    def forward(self, x1, *x2):
        x1 = self.up(x1)
        x = self.pad_and_concat(x1, *x2)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



