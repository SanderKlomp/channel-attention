# %% Get flops (and params)
import torch
import thop
import pprint
from models.resnet import resnet, Bottleneck, BasicBlock
from models.recalibration_modules import SRMLayer, SELayer, MeanReweightLayer, MultiSELayer, ECALayer

def count_meanrew(m, x, y):
    x = x[0]
    n_channels = x.shape[1]
    x_numel = x.numel()
    mean_ops = x_numel + n_channels
    cfc_ops = n_channels
    integrate_ops = x_numel
    m.total_ops += torch.DoubleTensor([int(cfc_ops) + int(mean_ops) + int(integrate_ops)])

def count_srm(m, x, y):
    x = x[0] # get rid of tuple. N x C x H x W
    n_channels = x.shape[1] # For each of the 2 CFC layers
    x_numel = x.numel()

    mean_ops = x_numel + n_channels # count the mean
    # count the std (assume mean is used, even though in reality it isn't)
    std_ops = x_numel * 3 + x_numel + n_channels # sqrt((x-mu)^2) -> 3*numel ops, then taking mean is numel+channels ops
    cfc_ops = 2 * n_channels + n_channels # includes summing the CFC results with each other
    bn_ops = 2 * n_channels
    integrate_ops = n_channels * 4 + x_numel # apply sigmoid and sum back to input

    m.total_ops += torch.DoubleTensor([int(cfc_ops) + int(mean_ops) + int(std_ops) + int(integrate_ops) + int(bn_ops)])

def count_se(m, x, y):
    x = x[0]
    n_channels = x.shape[1]
    x_numel = x.numel()
    mean_ops = x_numel + n_channels
    fc_ops = n_channels * n_channels * 2 # two fc layers
    integrate_ops = n_channels * 4 + x_numel
    m.total_ops += torch.DoubleTensor([int(fc_ops) + int(mean_ops) + int(integrate_ops)])

def count_multise(m, x, y):
    x = x[0]
    n_channels = x.shape[1]
    x_numel = x.numel()
    mean_ops = x_numel + n_channels
    fc_ops = n_channels * n_channels * 7 # three sets of two fc layers plus domain assignment FC layer = 7 FC layers
    integrate_ops = n_channels * 3 + n_channels + n_channels * 4 + x_numel # Softmax, multiply, sigmoid, multiply with x
    m.total_ops += torch.DoubleTensor([int(fc_ops) + int(mean_ops) + int(integrate_ops)])

def count_eca(m, x, y):
    x = x[0]
    n_channels = x.shape[1]
    x_numel = x.numel()
    mean_ops = x_numel + n_channels
    conv1d_ops = n_channels * 3 # Assume fixed k=3
    integrate_ops = x_numel + n_channels * 4
    m.total_ops += torch.DoubleTensor([int(conv1d_ops) + int(mean_ops) + int(integrate_ops)])

def count_resblock(m, x, y): # Original thop implementation does not account for the residual summation
    m.total_ops += torch.DoubleTensor([int(y.numel())])

def count_sigmoid(m, x, y):
    x = x[0]
    nelements = x.numel()
    m.total_ops += torch.DoubleTensor([int(nelements * 4)])

input = torch.randn(1, 3, 224, 224)
flop_dict = {} # Measures number of parameters in millions
for depth in [18, 34, 50, 101, 152]:
    for recalibration_type in ['eca']: #None, 'se', 'srm', 'meanrew', 'multise', 'eca'
        model_resnet = resnet(depth=depth, recalibration_type=recalibration_type).eval()
        flop_dict[(depth, recalibration_type)] = thop.profile(model_resnet, inputs=(input, ),
                                                              custom_ops={SRMLayer: count_srm,
                                                                          SELayer: count_se,
                                                                          MeanReweightLayer: count_meanrew,
                                                                          MultiSELayer: count_multise,
                                                                          ECALayer: count_eca})
pprint.pprint(flop_dict)

# %%
