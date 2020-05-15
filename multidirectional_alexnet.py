import torch.nn as nn
import torch
from torch.autograd import Variable
from models.cnn_feedforward_recurrent_topdown import MultiDirectional
from loss.crossentropy_plus import CrossEntropyLossPlus
import pandas as pd

bidirectional_alexnet_parameters = list()
# Layer 1
layer = dict()
layer['input_channels'] = 3
layer['output_channels'] = 96
layer['topdown_channels'] = 256
layer['input_size'] = [3, 227, 227]
layer['dropout'] = None
layer['type'] = 'conv'
ff_rc_params = dict()
ff_rc_params['activation'] = 'relu'
ff_rc_params['kernel_size'] = 11
ff_rc_params['stride'] = 4
ff_rc_params['padding'] = 0
layer['feedforward_recurrent_parameters'] = ff_rc_params
td_params = dict()
td_params['activation'] = 'relu'
td_params['kernel_size'] = 11 + (4 * (3 - 1)) + (4 * 2 * (5 - 1))
td_params['stride'] = 4 * 2 * 1
td_params['padding'] = 0 + (0 * 4) + (2 * 4 * 2)
layer['topdown_parameters'] = td_params
bidirectional_alexnet_parameters.append(layer)
# Layer 1 pool
layer = dict()
layer['type'] = 'pool'
layer['input_size'] = [96, 55, 55]
params = dict()
params['kernel_size'] = 3
params['stride'] = 2
params['padding'] = 0
layer['parameters'] = params
bidirectional_alexnet_parameters.append(layer)
# Layer 2
layer = dict()
layer['type'] = 'conv'
layer['dropout'] = None
layer['input_channels'] = 96
layer['output_channels'] = 256
layer['topdown_channels'] = 384
layer['input_size'] = [96, 27, 27]
ff_rc_params = dict()
ff_rc_params['activation'] = 'relu'
ff_rc_params['kernel_size'] = 5
ff_rc_params['stride'] = 1
ff_rc_params['padding'] = 2
layer['feedforward_recurrent_parameters'] = ff_rc_params
td_params = dict()
td_params['activation'] = 'relu'
td_params['kernel_size'] = 5 + (1 * (3 - 1)) + (1 * 2 * (3 - 1))
td_params['stride'] = 1 * 2 * 1
td_params['padding'] = 2 + (1 * 0) + (1 * 2 * 1)
layer['topdown_parameters'] = td_params
bidirectional_alexnet_parameters.append(layer)
# Layer 2 pool
layer = dict()
layer['type'] = 'pool'
layer['input_size'] = [256, 27, 27]
params = dict()
params['kernel_size'] = 3
params['stride'] = 2
params['padding'] = 0
layer['parameters'] = params
bidirectional_alexnet_parameters.append(layer)
# Layer 3
layer = dict()
layer['type'] = 'conv'
layer['input_channels'] = 256
layer['output_channels'] = 384
layer['topdown_channels'] = 384
layer['input_size'] = [256, 13, 13]
layer['dropout'] = None
ff_rc_params = dict()
ff_rc_params['activation'] = 'relu'
ff_rc_params['kernel_size'] = 3
ff_rc_params['stride'] = 1
ff_rc_params['padding'] = 1
layer['feedforward_recurrent_parameters'] = ff_rc_params
td_params = dict()
td_params['activation'] = 'relu'
td_params['kernel_size'] = 3 + (1 * (3 - 1))
td_params['stride'] = 1 * 1
td_params['padding'] = 1 + (1 * 1)
layer['topdown_parameters'] = td_params
bidirectional_alexnet_parameters.append(layer)
# Layer 4
layer = dict()
layer['type'] = 'conv'
layer['input_channels'] = 384
layer['output_channels'] = 384
layer['topdown_channels'] = 256
layer['input_size'] = [384, 13, 13]
layer['dropout'] = None
ff_rc_params = dict()
ff_rc_params['activation'] = 'relu'
ff_rc_params['kernel_size'] = 3
ff_rc_params['stride'] = 1
ff_rc_params['padding'] = 1
layer['feedforward_recurrent_parameters'] = ff_rc_params
td_params = dict()
td_params['activation'] = 'relu'
td_params['kernel_size'] = 3 + (1 * (3 - 1))
td_params['stride'] = 1 * 1
td_params['padding'] = 1 + (1 * 1)
layer['topdown_parameters'] = td_params
bidirectional_alexnet_parameters.append(layer)
# Layer 5
layer = dict()
layer['type'] = 'conv'
layer['input_channels'] = 384
layer['output_channels'] = 256
layer['topdown_channels'] = 4096
layer['input_size'] = [384, 13, 13]
layer['dropout'] = None
ff_rc_params = dict()
ff_rc_params['activation'] = 'relu'
ff_rc_params['kernel_size'] = 3
ff_rc_params['stride'] = 1
ff_rc_params['padding'] = 1
layer['feedforward_recurrent_parameters'] = ff_rc_params
td_params = dict()
td_params['activation'] = 'relu'
td_params['kernel_size'] = 13
td_params['stride'] = 1 * 1
td_params['padding'] = 0
layer['topdown_parameters'] = td_params
bidirectional_alexnet_parameters.append(layer)
# Layer 5 pool
layer = dict()
layer['type'] = 'pool'
layer['input_size'] = [256, 13, 13]
params = dict()
params['kernel_size'] = 3
params['stride'] = 2
params['padding'] = 0
layer['parameters'] = params
bidirectional_alexnet_parameters.append(layer)
# Layer 6
layer = dict()
layer['type'] = 'fc'
layer['input_channels'] = 9216
layer['output_channels'] = 4096
layer['topdown_channels'] = 4096
layer['input_size'] = [9216]
layer['dropout'] = 0.5
ff_rc_params = dict()
ff_rc_params['activation'] = 'relu'
layer['feedforward_recurrent_parameters'] = ff_rc_params
td_params = dict()
td_params['activation'] = 'relu'
layer['topdown_parameters'] = td_params
bidirectional_alexnet_parameters.append(layer)
# Layer 7
layer = dict()
layer['type'] = 'fc'
layer['input_channels'] = 4096
layer['output_channels'] = 4096
layer['topdown_channels'] = 1000
layer['input_size'] = [4096]
layer['dropout'] = 0.5
ff_rc_params = dict()
ff_rc_params['activation'] = 'relu'
layer['feedforward_recurrent_parameters'] = ff_rc_params
td_params = dict()
td_params['activation'] = 'relu'
layer['topdown_parameters'] = td_params
bidirectional_alexnet_parameters.append(layer)
# Layer 8
layer = dict()
layer['type'] = 'fc'
layer['input_channels'] = 4096
layer['output_channels'] = 1000
layer['topdown_channels'] = 1000
layer['input_size'] = [4096]
layer['dropout'] = None
ff_rc_params = dict()
ff_rc_params['activation'] = 'softmax'
layer['feedforward_recurrent_parameters'] = ff_rc_params
td_params = dict()
td_params['activation'] = 'relu'
layer['topdown_parameters'] = td_params
bidirectional_alexnet_parameters.append(layer)


# inout size
input_size = [3, 227, 227]
output_size = [1000]

module = MultiDirectional(input_size,
                          output_size,
                          bidirectional_alexnet_parameters,
                          timeout=1,
                          recognition_threshold=0.5)

x_ff = Variable(torch.randn(1, 3, 227, 227), requires_grad=True)
# x_ff.retain_grad()

out, rec, topdwn = module(x_ff)
from torchviz import make_dot
make_dot(module(x_ff), params=dict(module.named_parameters()))


loss_fn = CrossEntropyLossPlus()
target = torch.zeros(1, dtype=torch.long)
loss = loss_fn(out, rec, topdwn, target)
loss.backward()
out, rec, topdwn = module(x_ff)
loss = loss_fn(out, rec, topdwn, target)
loss.backward()


print(x_ff.grad)