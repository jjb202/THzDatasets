'''
HINet: Half Instance Normalization Network for Image Restoration

@inproceedings{chen2021hinet,
  title={HINet: Half Instance Normalization Network for Image Restoration},
  author={Liangyu Chen and Xin Lu and Jie Zhang and Xiaojie Chu and Chengpeng Chen},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2021}
}
'''

import torch
import torch.nn as nn
from torch.nn import Softmax
# from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torch.autograd._functions import Resize

import torch.nn.functional as F
def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)



## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class PixelShuffle_Down(nn.Module):
    def __init__(self, scale=2):
        super(PixelShuffle_Down, self).__init__()
        self.scale = scale
    def forward(self, x):
        # assert h%scale==0 and w%scale==0
        b,c,h,w = x.size()
        x = x[:,:,:int(h-h%self.scale), :int(w-w%self.scale)]
        out_c = c*(self.scale**2)
        out_h = h//self.scale
        out_w = w//self.scale
        out = x.contiguous().view(b, c, out_h, self.scale, out_w, self.scale)
        return out.permute(0,1,3,5,2,4).contiguous().view(b, out_c, out_h, out_w)

class HINet(nn.Module):

    def __init__(self, in_chn=3, wf=64, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(HINet, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.down_path_3 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_03 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth): #0,1,2,3,4
            use_HIN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN))
            self.down_path_2.append(UNetConv4Block(prev_channels, (2**i) * wf, downsample, relu_slope, use_csff=downsample, use_HIN=use_HIN))
            self.down_path_3.append(UNetConv4Block(prev_channels, (2**i) * wf, downsample, relu_slope, use_csff=downsample, use_HIN=use_HIN))
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.up_path_3 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        self.skip_conv_3 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_2.append(UNetUp4Block(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_3.append(UNetUp4Block(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_3.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)
        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        # self.non = CrissCrossAttention(prev_channels)
        # self.sima = simam_module()
        self.last = conv3x3(prev_channels, in_chn, bias=True)
        self.upsamp = nn.Upsample(size=(32, 32), mode='nearest')
        self.upsamp1 = nn.Upsample(size=(64, 64), mode='nearest')
        self.upsamp2 = nn.Upsample(size=(128, 128), mode='nearest')
        self.upsamp3 = nn.Upsample(size=(256, 256), mode='nearest')
        self.con1 = conv3x3(960, 512, bias=True)
        self.con2 = conv3x3(960, 256, bias=True)
        self.con3 = conv3x3(960, 128, bias=True)
        self.con4 = conv3x3(960, 64, bias=True)

        self.upsamp_0 = nn.Upsample(size=(32, 32), mode='nearest')
        self.upsamp1_1 = nn.Upsample(size=(64, 64), mode='nearest')
        self.upsamp2_2 = nn.Upsample(size=(128, 128), mode='nearest')
        self.upsamp3_3 = nn.Upsample(size=(256, 256), mode='nearest')
        self.con1_1 = conv3x3(960, 512, bias=True)
        self.con2_2 = conv3x3(960, 256, bias=True)
        self.con3_3 = conv3x3(960, 128, bias=True)
        self.con4_4 = conv3x3(960, 64, bias=True)

        self.upsamp_0_0 = nn.Upsample(size=(32, 32), mode='nearest')
        self.upsamp1_1_1 = nn.Upsample(size=(64, 64), mode='nearest')
        self.upsamp2_2_2 = nn.Upsample(size=(128, 128), mode='nearest')
        self.upsamp3_3_3 = nn.Upsample(size=(256, 256), mode='nearest')
        self.con1_1_1 = conv3x3(960, 512, bias=True)
        self.con2_2_2 = conv3x3(960, 256, bias=True)
        self.con3_3_3 = conv3x3(960, 128, bias=True)
        self.con4_4_4 = conv3x3(960, 64, bias=True)

    def forward(self, x):
        image = x

        #stage 1
        x1 = self.conv_01(image)

        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)

        for i, up in enumerate(self.up_path_1):
            if i == 0:
                bridge = self.skip_conv_1[i](encs[-i-1])
                bridge2 = self.upsamp(self.skip_conv_1[i+1](encs[-i-2]))
                bridge3 = self.upsamp(self.skip_conv_1[i+2](encs[-i-3]))
                bridge4 = self.upsamp(self.skip_conv_1[i+3](encs[-i-4]))
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con1(out)
                x1 = up(x1, out)
            if i == 1:
                bridge = self.upsamp1(self.skip_conv_1[i-1](encs[-i]))
                bridge2 = self.skip_conv_1[i](encs[-i-1])
                bridge3 = self.upsamp1(self.skip_conv_1[i+1](encs[-i-2]))
                bridge4 = self.upsamp1(self.skip_conv_1[i+2](encs[-i-3]))
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con2(out)
                x1 = up(x1, out)
            if i == 2:
                bridge = self.upsamp2(self.skip_conv_1[i-2](encs[-i+1]))
                bridge2 = self.upsamp2(self.skip_conv_1[i-1](encs[-i]))
                bridge3 = self.skip_conv_1[i](encs[-i-1])
                bridge4 = self.upsamp2(self.skip_conv_1[i+1](encs[-i-2]))
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con3(out)
                x1 = up(x1, out)
            if i == 3:
                bridge = self.upsamp3(self.skip_conv_1[i-3](encs[-i+2]))
                bridge2 = self.upsamp3(self.skip_conv_1[i-2](encs[-i+1]))
                bridge3 = self.upsamp3(self.skip_conv_1[i-1](encs[-i]))
                bridge4 = self.skip_conv_1[i](encs[-i-1])
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con4(out)
                x1 = up(x1, out)
            # x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)

        sam_feature, out_1 = self.sam12(x1, image)
        #stage 2
        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        # x2 = self.non(x2)
        # x2 = self.sima(x2)
        enc2 = []
        decs2 = []
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i-1])
                enc2.append(x2_up)
            else:
                x2 = down(x2)

        for i, up in enumerate(self.up_path_2):
            if i == 0:
                bridge = self.skip_conv_2[i](enc2[-i-1])
                bridge2 = self.upsamp_0(self.skip_conv_2[i+1](enc2[-i-2]))
                bridge3 = self.upsamp_0(self.skip_conv_2[i+2](enc2[-i-3]))
                bridge4 = self.upsamp_0(self.skip_conv_2[i+3](enc2[-i-4]))
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con1_1(out)
                x2 = up(x2, out)
            if i == 1:
                bridge = self.upsamp1_1(self.skip_conv_2[i-1](enc2[-i]))
                bridge2 = self.skip_conv_2[i](enc2[-i-1])
                bridge3 = self.upsamp1_1(self.skip_conv_2[i+1](enc2[-i-2]))
                bridge4 = self.upsamp1_1(self.skip_conv_2[i+2](enc2[-i-3]))
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con2_2(out)
                x2 = up(x2, out)
            if i == 2:
                bridge = self.upsamp2_2(self.skip_conv_2[i-2](enc2[-i+1]))
                bridge2 = self.upsamp2_2(self.skip_conv_2[i-1](enc2[-i]))
                bridge3 = self.skip_conv_2[i](enc2[-i-1])
                bridge4 = self.upsamp2_2(self.skip_conv_2[i+1](enc2[-i-2]))
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con3_3(out)
                x2 = up(x2, out)
            if i == 3:
                bridge = self.upsamp3_3(self.skip_conv_2[i-3](enc2[-i+2]))
                bridge2 = self.upsamp3_3(self.skip_conv_2[i-2](enc2[-i+1]))
                bridge3 = self.upsamp3_3(self.skip_conv_2[i-1](enc2[-i]))
                bridge4 = self.skip_conv_2[i](enc2[-i-1])
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con4_4(out)
                x2 = up(x2, out)
            # x2 = up(x2, self.skip_conv_2[i](enc2[-i-1]))
            decs2.append(x2)
        sam_feature2, out_2 = self.sam12(x2, image)
        #stage 3
        x3 = self.conv_03(image)
        x3 = self.cat12(torch.cat([x3, sam_feature2], dim=1))
        # x2 = self.non(x2)
        # x2 = self.sima(x2)
        enc3 = []

        for i, down in enumerate(self.down_path_3):
            if (i+1) < self.depth:
                x3, x3_up = down(x3, enc2[i], decs2[-i-1])
                enc3.append(x3_up)
            else:
                x3 = down(x3)

        for i, up in enumerate(self.up_path_3):
            if i == 0:
                bridge = self.skip_conv_3[i](enc3[-i-1])
                bridge2 = self.upsamp_0_0(self.skip_conv_3[i+1](enc3[-i-2]))
                bridge3 = self.upsamp_0_0(self.skip_conv_3[i+2](enc3[-i-3]))
                bridge4 = self.upsamp_0_0(self.skip_conv_3[i+3](enc3[-i-4]))
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con1_1_1(out)
                x3 = up(x3, out)
            if i == 1:
                bridge = self.upsamp1_1_1(self.skip_conv_3[i-1](enc3[-i]))
                bridge2 = self.skip_conv_3[i](enc3[-i-1])
                bridge3 = self.upsamp1_1_1(self.skip_conv_3[i+1](enc3[-i-2]))
                bridge4 = self.upsamp1_1_1(self.skip_conv_3[i+2](enc3[-i-3]))
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con2_2_2(out)
                x3 = up(x3, out)
            if i == 2:
                bridge = self.upsamp2_2_2(self.skip_conv_3[i-2](enc3[-i+1]))
                bridge2 = self.upsamp2_2_2(self.skip_conv_3[i-1](enc3[-i]))
                bridge3 = self.skip_conv_3[i](enc3[-i-1])
                bridge4 = self.upsamp2_2_2(self.skip_conv_3[i+1](enc3[-i-2]))
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con3_3_3(out)
                x3 = up(x3, out)
            if i == 3:
                bridge = self.upsamp3_3_3(self.skip_conv_3[i-3](enc3[-i+2]))
                bridge2 = self.upsamp3_3_3(self.skip_conv_3[i-2](enc3[-i+1]))
                bridge3 = self.upsamp3_3_3(self.skip_conv_3[i-1](enc3[-i]))
                bridge4 = self.skip_conv_3[i](enc3[-i-1])
                out = torch.cat([bridge, bridge2, bridge3, bridge4], 1)
                out = self.con4_4_4(out)
                x3 = up(x3, out)
            # x3 = up(x3, self.skip_conv_3[i](blocks3[-i-1]))

        out_3 = self.last(x3)
        out_3 = out_3 + image

        return [out_1, out_2, out_3]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetUp4Block(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUp4Block, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConv4Block(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class UNetConv4Block(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConv4Block, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            # self.reluDes1 = nn.PReLU()
            # self.reluDes2 = nn.PReLU()
            # self.covDes1 = nn.Conv2d(2*out_size, out_size, 3, 1, 1)
            #
            # self.reluDes1_1 = nn.PReLU()
            # self.reluDes2_2 = nn.PReLU()
            # self.covDes1_1 = nn.Conv2d(2*out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_size//4, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        self.conv1 = nn.Conv2d(out_size//4, out_size//4, 3, 1, 1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(2*out_size//4, out_size//4, 3, 1, 1)
        self.relu2 = nn.PReLU()


    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2, out_3, out_4 = torch.chunk(out, 4, dim=1)

            out3 = self.relu1(self.conv1(out_3))
            out3 = torch.cat([out_3, out3], 1)
            out3 = self.conv2(out3)
            out3 += out_3
            out3 = self.relu2(out3)

            out4 = self.relu1(self.conv1(out_4))
            out4 = torch.cat([out_4, out4], 1)
            out4 = self.conv2(out4)
            out4 += out_4
            out4 = self.relu2(out4)
            out = torch.cat([self.norm2(out_1), out_2, out3, out4], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            # encO = self.reluDes1(self.csff_enc(enc))
            # enc0 = torch.cat([enc, encO],1)
            # enc0 = self.covDes1(enc0)
            # enc0 += enc
            # enc0 = self.reluDes2(enc0)
            #
            # decO = self.reluDes1_1(self.csff_dec(dec))
            # decO = torch.cat([dec, decO],1)
            # decO = self.covDes1_1(decO)
            # decO += dec
            # decO = self.reluDes2_2(decO)
            # out = out + enc0 + decO
            # default code
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc


if __name__ == "__main__":
    pass
