"""Converts an imagenet checkpoint that may be unreadable 
using torch.load in older version of PyTorch to something that IS readable."""
#%% Load original model
import os
from collections import OrderedDict
import torch
from models.resnet import resnet
checkpoint_path = "/home/sander/Desktop/model_best.pth.tar"
save_path = os.path.splitext(checkpoint_path)[0]
recalibration_type = 'se'
model = resnet(depth=50,
               recalibration_type=recalibration_type,
              )


model = torch.nn.DataParallel(model).cuda()
print(hasattr(model, 'module'))


checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint['state_dict'])
# %% Do the actual conversion
if hasattr(model, 'module'):
    model = model.module

def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on CPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu

checkpoint_cpu = {}
checkpoint_cpu['state_dict'] = weights_to_cpu(model.state_dict())

torch.save(checkpoint_cpu['state_dict'], save_path, _use_new_zipfile_serialization=False)

# %%
