import math

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        # channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        # channel_std = channel_var.sqrt()
        channel_std = x.view(N, C, -1).std(dim=2, keepdim=True)

        t = torch.cat((channel_mean, channel_std), dim=2)
        return t

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.Sigmoid()

        self.reduction = reduction

        self.fc = nn.Sequential(
                nn.Linear(channel, channel // self.reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // self.reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # avg_y = self.avgpool(x).view(b, c)
        avg_y = torch.mean(x, dim=(2,3))

        gate = self.fc(avg_y).view(b, c, 1, 1)
        gate = self.activation(gate)

        return x * gate


class GELayer(nn.Module):
    def __init__(self, channel, layer_idx):
        super(GELayer, self).__init__()

        # Kernel size w.r.t each layer for global depth-wise convolution
        kernel_size = [-1, 56, 28, 14, 7][layer_idx]

        self.conv = nn.Sequential(
                        nn.Conv2d(channel, channel, kernel_size=kernel_size, groups=channel), 
                        nn.BatchNorm2d(channel),
                    )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        gate = self.conv(x)
        gate = self.activation(gate)

        return x * gate

class MeanReweightLayer(nn.Module):
    """Renamed to Attention-Bias (AB) layer in paper"""
    def __init__(self, channel):
        super(MeanReweightLayer, self).__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cfc = Parameter(torch.Tensor(channel))
        self.cfc.data.fill_(0)

    def forward(self, x):
        # avg_y = self.avgpool(x)  # B x C
        avg_y = torch.mean(x, dim=(2,3), keepdim=True) # This is only half as computationally expensive as adaptive avg pooling
        avg_y = avg_y * self.cfc[None, :, None, None]
        return x + avg_y

class MultiSELayer(nn.Module):
    def __init__(self, channel, reduction=16, num_branches=3):
        super(MultiSELayer, self).__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.Sigmoid()

        self.reduction = reduction
        self.num_branches = num_branches
        self.channel = channel
        new_channel = channel * num_branches
        self.fc = nn.Sequential(nn.Conv2d(new_channel, new_channel // self.reduction, kernel_size=1, bias=True, groups=num_branches),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(new_channel // self.reduction, new_channel, kernel_size=1, bias=True, groups=num_branches))

        self.style_assigner = nn.Linear(channel, num_branches, bias=False)
    
    def _style_assignment(self, channel_mean):
        N, _, _, _ = channel_mean.size()
        style_assignment = self.style_assigner(channel_mean.view(N, -1))
        style_assignment = nn.functional.softmax(style_assignment, dim=1)
        return style_assignment

    def forward(self, x):
        # avg_y = self.avgpool(x) # B x C x 1 x 1
        avg_y = torch.mean(x, dim=(2,3), keepdim=True) # B x C x 1 x 1
        B, C, _, _ = avg_y.shape
        style_assignment = self._style_assignment(avg_y) # B x N

        avg_y = avg_y.repeat(1, self.num_branches, 1, 1) # B x NC x 1 x 1
        z = self.fc(avg_y) # B x NC x 1 x 1
        style_assignment = style_assignment.repeat_interleave(C, dim=1)

        z = z * style_assignment[:, :, None, None]
        z = torch.sum(z.view(B, self.num_branches, C, 1, 1), dim=1) # B x C x 1 x 1
        z = self.activation(z)

        return x * z

class Multi3SELayer(nn.Module):
    """Hardcoded variant of the MultiSELayer with 3 branches, which is convertable to TensorRt"""
    def __init__(self, channel, reduction=16):
        super(Multi3SELayer, self).__init__()
        self.activation = nn.Sigmoid()

        self.reduction = reduction

        self.fc1 = nn.Sequential(
                nn.Linear(channel, channel // self.reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // self.reduction, channel),
                
        )
        self.fc2 = nn.Sequential(
                nn.Linear(channel, channel // self.reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // self.reduction, channel),
        )
        self.fc3 = nn.Sequential(
                nn.Linear(channel, channel // self.reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // self.reduction, channel),
        )
        self.style_assigner = nn.Linear(channel, 3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_y = torch.mean(x, dim=(2,3)) # B x C

        res1 = self.fc1(avg_y).view(b, c, 1)
        res2 = self.fc2(avg_y).view(b, c, 1)
        res3 = self.fc3(avg_y).view(b, c, 1)
        res = torch.cat((res1, res2, res3), dim=2) # B x C x 3

        domain_assignment = self.style_assigner(avg_y)
        domain_assignment = nn.functional.softmax(domain_assignment, dim=1).view(-1, 1, 3)

        res = res * domain_assignment # B x C x 3
        res = torch.sum(res, dim=2).view(b, c, 1, 1)
        res = self.activation(res)

        return x * res

# class MultiSRMLayer(nn.Module):
#     def __init__(self, channel, num_branches=8):
#         super(MultiSRMLayer, self).__init__()

#         # SRM
#         self.cfc = Parameter(torch.Tensor(channel, 2, num_branches))
#         self.cfc.data.fill_(0)
#         self.num_branches = num_branches
#         self.bn = nn.BatchNorm2d(channel)
#         self.activation = nn.Sigmoid()

#         setattr(self.cfc, 'srm_param', True)
#         setattr(self.bn.weight, 'srm_param', True)
#         setattr(self.bn.bias, 'srm_param', True)

#         # Style assignment
#         self.style_assigner = nn.Linear(channel * 2, num_branches, bias=False)  # *2, because mean and variance

#     def _style_pooling(self, x, eps=1e-5):
#         N, C, _, _ = x.size()

#         channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
#         # channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
#         # channel_std = channel_var.sqrt()
#         channel_std = x.view(N, C, -1).std(dim=2, keepdim=True)

#         channel_meanstd = torch.cat((channel_mean, channel_std), dim=2)  # B x C x 2

#         return channel_meanstd

#     def _style_integration(self, channel_meanstd, style_assignment):
#         z = channel_meanstd.unsqueeze(-1).expand(-1, -1, -1, self.num_branches)  # B x C x 2 x num_branches
#         z = z * self.cfc[None, :, :, :]
#         z = torch.sum(z, dim=2)  # B x C x num_branches

#         # Scale the activations per branch
#         z = z * style_assignment[:, None, :]
#         z = torch.sum(z, dim=2)

#         z_hat = self.bn(z[:, :, None, None])

#         g = self.activation(z_hat)

#         return g

#     def _style_assignment(self, channel_meanstd):
#         N, _, _ = channel_meanstd.size()
#         style_assignment = self.style_assigner(channel_meanstd.view(N, -1))  # unpack mean/variance into single tensor
#         style_assignment = nn.functional.softmax(style_assignment, dim=1)
#         return style_assignment

#     def forward(self, x):
#         channel_meanstd = self._style_pooling(x)  # B x C x 2

#         # B x C x 1 x 1
#         style_assignment = self._style_assignment(channel_meanstd)
#         g = self._style_integration(channel_meanstd, style_assignment)

#         return x * g


# class NALayer(nn.Module):
#     def __init__(self, in_channel):
#         super(NALayer, self).__init__()
#         self.sig = nn.Sigmoid()

#         self.weight2 = Parameter(torch.zeros(1))
#         self.bias2 = Parameter(torch.ones(1))


#     def forward(self, x):
#         b, c, h, w = x.size()

#         # Context Modeling
#         x_context = torch.mean(x, 1, keepdim=True) # mean over channel dimension
#         x_context = x_context.view(b, 1, h * w, 1)
#         x_diff = -abs(x_context - x_context.mean(dim=2, keepdim=True)) # subtract spatial mean
#         x_diff = F.softmax(x_diff, dim=2)
#         x_new = x.view(b, 1, c, h * w)
#         context = torch.matmul(x_new, x_diff)
#         context = context.view(b, c)

#         # Normalization
#         x_channel = context - context.mean(dim=1, keepdim=True)
#         std = x_channel.std(dim=1, keepdim=True) + 1e-5
#         x_channel = x_channel / std
#         x_channel = x_channel.view(b, c, 1, 1)
#         x_channel = x_channel * self.weight2 + self.bias2
#         x_channel = self.sig(x_channel)
#         x = x * x_channel

#         return x

class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = torch.mean(x, dim=(2,3), keepdim=True) #b x c x 1 x 1

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # b x c x 1 x 1?

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y