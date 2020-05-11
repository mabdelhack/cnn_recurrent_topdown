from models.triple_layer import TripleLayerConv
import torch.nn as nn
import torch
from torch.autograd import Variable
import pandas as pd

input_channels = 3
output_channels = 96
topdown_channels = 256
feedforward_recurrent_parameters = dict()
feedforward_recurrent_parameters['kernel_size'] = 11
feedforward_recurrent_parameters['stride'] = 4
feedforward_recurrent_parameters['padding'] = 0
feedforward_recurrent_parameters['activation'] = 'relu'

topdown_parameters = dict()
topdown_parameters['kernel_size'] = 11 + 5 * 3 - 3
topdown_parameters['stride'] = 8
topdown_parameters['padding'] = 2
topdown_parameters['activation'] = 'relu'


module = TripleLayerConv(input_channels, output_channels, topdown_channels, feedforward_recurrent_parameters,
                     topdown_parameters)

x_ff = Variable(torch.randn(1, 3, 227, 227), requires_grad=True)
x_rc = Variable(torch.zeros(1, 96, 55, 55), requires_grad=True)
x_td = Variable(torch.randn(1, 256, 27, 27), requires_grad=True)
x_ff.retain_grad()
x_rc.retain_grad()
x_td.retain_grad()

out, rec, topdwn = module([x_ff, x_rc, x_td], feedforward_flag=True, recurrent_flag=False, topdown_flag=False)
out, rec, topdwn = module([x_ff, out, x_td])
out, rec, topdwn = module([x_ff, out, x_td])
out, rec, topdwn = module([x_ff, out, x_td])
loss_fn = nn.MSELoss()
loss = loss_fn(out, x_rc)
loss.backward()

print(x_ff.grad)