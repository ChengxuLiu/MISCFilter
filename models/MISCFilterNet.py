import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
from torch.nn.utils import weight_norm
import models.MISCKernel_cuda as misckernel

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=ResBlock):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)

class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-inchannel, kernel_size=1, stride=1, relu=True)
        )
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)
    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)
    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data ** 2 + epsilon ** 2))

def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device

    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype),
            indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.type(x.type())
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


class MISCKernelNet(nn.Module):
    def __init__(self,
                inp_channels=3,
                out_channels=3,
                dim=32,
                num_blocks=[12,12,12],
                num_blocks_kernel=[1,1,1],
                kernel_size=7,
                inference=False,
                ):
        super(MISCKernelNet, self).__init__()
        self.inference = inference
        self.dim = dim
        self.kernel_size = kernel_size
        self.kernel_pad = int((self.kernel_size - 1) / 2.0)

        if not inference:
            BasicConv = BasicConv_do
            ResBlock = ResBlock_do_fft_bench
        else:
            BasicConv = BasicConv_do_eval
            ResBlock = ResBlock_do_fft_bench_eval
        base_channel = dim

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_blocks[0], ResBlock=ResBlock),
            EBlock(base_channel*2, num_blocks[1], ResBlock=ResBlock),
            EBlock(base_channel*4, num_blocks[2], ResBlock=ResBlock),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(inp_channels, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4*2, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_blocks[2], ResBlock=ResBlock),
            DBlock(base_channel * 2, num_blocks[1], ResBlock=ResBlock),
            DBlock(base_channel, num_blocks[0], ResBlock=ResBlock)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel*2, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

        self.softmax = nn.Softmax(1)
        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
        self.moduleKernel = misckernel.FunctionKernel.apply

        self.KernelPredictFlow = nn.ModuleList([
                BasicConv(base_channel * 4, 2, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 2, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel, 2, kernel_size=3, relu=False, stride=1),
        ])
        self.flowup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.KernelPredictFlowMask = nn.ModuleList([
                BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1),
        ])
        self.sigmoid = nn.Sigmoid()


        self.KernelOutBias = nn.ModuleList([
                BasicConv(base_channel * 4, out_channels, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, out_channels, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel, out_channels, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutWeight = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutkernelx = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutkernely = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutAlpha = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutBeta = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])


    def forward(self, x):
        
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs_fil = list()
        outputs = list()
        Kernal_Loss = 0

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        

        s3_kernal_flow = self.KernelPredictFlow[0](z)
        s3_kernal_flowmask = self.KernelPredictFlowMask[0](z)
        s3_kernal_flowmask = self.sigmoid(s3_kernal_flowmask)

        zx4 = torch.cat([z,x_4],1)
        s3_kernal_flowfeat0,x_4_0  = torch.split(flow_warp(zx4, s3_kernal_flow.permute(0, 2, 3, 1)), self.dim * 4 , dim=1)
        s3_kernal_flowfeat1,x_4_1 = torch.split(flow_warp(zx4, -s3_kernal_flow.permute(0, 2, 3, 1)), self.dim * 4 , dim=1)
        x_4 = x_4_0 * s3_kernal_flowmask + x_4_1 * (1-s3_kernal_flowmask)

        s3_kernal_bias = self.KernelOutBias[0](z)

        z = torch.cat([z, s3_kernal_flowfeat0 * s3_kernal_flowmask + s3_kernal_flowfeat1 * (1-s3_kernal_flowmask)],1)
        s3_kernal_weight = self.KernelOutWeight[0](z)
        s3_kernal_weight = self.softmax(s3_kernal_weight)
        s3_kernal_alpha = self.KernelOutAlpha[0](z)
        s3_kernal_beta = self.KernelOutBeta[0](z)
        s3_kernal_posx = self.KernelOutkernelx[0](z)
        s3_kernal_posy = self.KernelOutkernely[0](z)
        z = self.feat_extract[3](z)

        out3 = self.moduleKernel(self.modulePad(  torch.cat([x_4,x_4.new_ones(x_4.size(0),1,x_4.size(2),x_4.size(3))], 1)  ),s3_kernal_posx,s3_kernal_posy, s3_kernal_alpha, s3_kernal_beta,s3_kernal_weight) 
        out3_norm = out3[:,-1:,:,:]
        out3_norm[out3_norm.abs()<0.01] = 1.0
        out3 = out3[:,:-1,:,:] / out3_norm 
        out3 += s3_kernal_bias
        if not self.inference:
            outputs.append(out3)
            outputs_fil.append(x_4)

            s3_Alpha = torch.mean(s3_kernal_weight * s3_kernal_alpha, dim=1, keepdim=True)
            s3_Beta = torch.mean(s3_kernal_weight * s3_kernal_beta, dim=1, keepdim=True)
            loss_s3_Alpha = CharbonnierFunc(s3_Alpha[:, :, :, :-1] - s3_Alpha[:, :, :, 1:]) + CharbonnierFunc(s3_Alpha[:, :, :-1, :] - s3_Alpha[:, :, 1:, :])
            loss_s3_Beta = CharbonnierFunc(s3_Beta[:, :, :, :-1] - s3_Beta[:, :, :, 1:]) + CharbonnierFunc(s3_Beta[:, :, :-1, :] - s3_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s3_Alpha
            Kernal_Loss += loss_s3_Beta

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        s2_kernal_flow = self.KernelPredictFlow[1](z) + self.flowup(s3_kernal_flow)*2
        s2_kernal_flowmask = self.KernelPredictFlowMask[1](z)
        s2_kernal_flowmask = self.sigmoid(s2_kernal_flowmask)

        zx2 = torch.cat([z,x_2],1)
        s2_kernal_flowfeat0,x_2_0  = torch.split(flow_warp(zx2, s2_kernal_flow.permute(0, 2, 3, 1)), self.dim * 2 , dim=1)
        s2_kernal_flowfeat1,x_2_1 = torch.split(flow_warp(zx2, -s2_kernal_flow.permute(0, 2, 3, 1)), self.dim * 2 , dim=1)
        x_2 = x_2_0 * s2_kernal_flowmask + x_2_1 * (1-s2_kernal_flowmask)

        s2_kernal_bias = self.KernelOutBias[1](z)

        z = torch.cat([z, s2_kernal_flowfeat0 * s2_kernal_flowmask + s2_kernal_flowfeat1 * (1-s2_kernal_flowmask)],1)
        s2_kernal_weight = self.KernelOutWeight[1](z)
        s2_kernal_weight = self.softmax(s2_kernal_weight)
        s2_kernal_alpha = self.KernelOutAlpha[1](z)
        s2_kernal_beta = self.KernelOutBeta[1](z)
        s2_kernal_posx = self.KernelOutkernelx[1](z)
        s2_kernal_posy = self.KernelOutkernely[1](z)
        z = self.feat_extract[4](z)

        out2 = self.moduleKernel(self.modulePad(  torch.cat([x_2,x_2.new_ones(x_2.size(0),1,x_2.size(2),x_2.size(3))], 1)  ), s2_kernal_posx,s2_kernal_posy, s2_kernal_alpha, s2_kernal_beta,s2_kernal_weight)
        out2_norm = out2[:,-1:,:,:]
        out2_norm[out2_norm.abs()<0.01] = 1.0
        out2 = out2[:,:-1,:,:] / out2_norm 
        out2 += s2_kernal_bias
        if not self.inference:
            outputs.append(out2)
            outputs_fil.append(x_2)

            s2_Alpha = torch.mean(s2_kernal_weight * s2_kernal_alpha, dim=1, keepdim=True)
            s2_Beta = torch.mean(s2_kernal_weight * s2_kernal_beta, dim=1, keepdim=True)
            loss_s2_Alpha = CharbonnierFunc(s2_Alpha[:, :, :, :-1] - s2_Alpha[:, :, :, 1:]) + CharbonnierFunc(s2_Alpha[:, :, :-1, :] - s2_Alpha[:, :, 1:, :])
            loss_s2_Beta = CharbonnierFunc(s2_Beta[:, :, :, :-1] - s2_Beta[:, :, :, 1:]) + CharbonnierFunc(s2_Beta[:, :, :-1, :] - s2_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s2_Alpha
            Kernal_Loss += loss_s2_Beta

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)

        z = self.Decoder[2](z)

        s1_kernal_flow = self.KernelPredictFlow[2](z) + self.flowup(s2_kernal_flow)*2
        s1_kernal_flowmask = self.KernelPredictFlowMask[2](z)
        s1_kernal_flowmask = self.sigmoid(s1_kernal_flowmask)

        zx = torch.cat([z,x],1)
        s1_kernal_flowfeat0,x_0  = torch.split(flow_warp(zx, s1_kernal_flow.permute(0, 2, 3, 1)), self.dim , dim=1)
        s1_kernal_flowfeat1,x_1 = torch.split(flow_warp(zx, -s1_kernal_flow.permute(0, 2, 3, 1)), self.dim , dim=1)
        x = x_0 * s1_kernal_flowmask + x_1 * (1-s1_kernal_flowmask)

        s1_kernal_bias = self.KernelOutBias[2](z)
        z = torch.cat([z, s1_kernal_flowfeat0 * s1_kernal_flowmask + s1_kernal_flowfeat1 * (1-s1_kernal_flowmask)],1)
        s1_kernal_weight = self.KernelOutWeight[2](z)
        s1_kernal_weight = self.softmax(s1_kernal_weight)
        s1_kernal_alpha = self.KernelOutAlpha[2](z)
        s1_kernal_beta = self.KernelOutBeta[2](z)
        s1_kernal_posx = self.KernelOutkernelx[2](z)
        s1_kernal_posy = self.KernelOutkernely[2](z)

        out = self.moduleKernel(self.modulePad(torch.cat([x,x.new_ones(x.size(0),1,x.size(2),x.size(3))], 1) ), s1_kernal_posx,s1_kernal_posy, s1_kernal_alpha, s1_kernal_beta, s1_kernal_weight)
        out_norm = out[:,-1:,:,:]
        out_norm[out_norm.abs()<0.01] = 1.0
        out = out[:,:-1,:,:] / out_norm 
        out += s1_kernal_bias
        if not self.inference:
            outputs.append(out)
            outputs_fil.append(x)

            s1_Alpha = torch.mean(s1_kernal_weight * s1_kernal_alpha, dim=1, keepdim=True)
            s1_Beta = torch.mean(s1_kernal_weight * s1_kernal_beta, dim=1, keepdim=True)
            loss_s1_Alpha = CharbonnierFunc(s1_Alpha[:, :, :, :-1] - s1_Alpha[:, :, :, 1:]) + CharbonnierFunc(s1_Alpha[:, :, :-1, :] - s1_Alpha[:, :, 1:, :])
            loss_s1_Beta = CharbonnierFunc(s1_Beta[:, :, :, :-1] - s1_Beta[:, :, :, 1:]) + CharbonnierFunc(s1_Beta[:, :, :-1, :] - s1_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s1_Alpha
            Kernal_Loss += loss_s1_Beta

            return outputs[::-1],outputs_fil[::-1]
        else:
            return out