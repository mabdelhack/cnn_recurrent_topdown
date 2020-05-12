import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from models.triple_layer import TripleLayerConv, TripleLayerFc
from models.pooling_layer import Pooling


class MultiDirectional(nn.Module):
    def __init__(self, input_size, output_size, layers_parameters, timeout=1, recognition_threshold=0.75, batch_size=1):
        """"""

        super(MultiDirectional, self).__init__()

        # Just some sanity checks
        if timeout < 1:
            raise ValueError("Timeout has to be greater than zero to at least run through the network once")
        if recognition_threshold <= 0 or recognition_threshold > 1:
            raise ValueError('Recognition threshold must be greater than zero and less than or equal one')

        # Model parameters based on inputs
        self.input_size = input_size
        self.output_size = output_size
        self.timeout = timeout
        self.threshold = recognition_threshold

        # Model intermediate inputs/outputs
        self.recurrent_inputs = list()
        self.topdown_inputs = list()
        self.recurrent_outputs = list()
        self.topdown_outputs = list()

        # Model layers
        self.layers = nn.ModuleList()
        for layer in layers_parameters:
            if layer['type'] == 'conv':
                module = TripleLayerConv(layer['input_channels'],
                                         layer['output_channels'],
                                         layer['topdown_channels'],
                                         layer['feedforward_recurrent_parameters'],
                                         layer['topdown_parameters'],
                                         dropout=layer['dropout'])

            elif layer['type'] == 'fc':
                module = TripleLayerFc(layer['input_channels'],
                                       layer['output_channels'],
                                       layer['topdown_channels'],
                                       layer['feedforward_recurrent_parameters'],
                                       layer['topdown_parameters'],
                                       dropout=layer['dropout'])

            elif layer['type'] == 'pool':
                module = Pooling(layer['parameters'])

            self.recurrent_inputs.append(torch.tensor(torch.zeros([batch_size] + layer['input_size'])))
            self.topdown_inputs.append(torch.tensor(torch.zeros([batch_size] + layer['input_size'])))
            self.recurrent_outputs.append(torch.tensor(torch.zeros([batch_size] + module.output_size(layer['input_size']))))
            self.topdown_outputs.append(torch.tensor(torch.zeros([batch_size] + module.output_size(layer['input_size']))))
            self.layers.append(module)

    def forward(self, input_data, topdown_input=None,
                feedforward_flag=True, recurrent_flag=True, topdown_flag=True, data_size=None):
        """"""
        # initial output (zeros to make sure we don't cross identification threshold before doing any processing)
        output = Variable(torch.zeros(self.output_size))
        if input_data.is_cuda:
            output = output.cuda()

        # topdown signal to the last layer (prior knowledge)
        if topdown_input is not None:
            x_topdown = topdown_input
        else:
            x_topdown = Variable(torch.zeros(self.output_size))
            if input_data.is_cuda:
                x_topdown = x_topdown.cuda()
        self.topdown_outputs.append(x_topdown)

        t = 0
        while t < self.timeout and torch.max(output).item() < self.threshold:
            # initial input
            x = input_data
            for idx, layer in enumerate(self.layers):
                next_layer = 2 if (len(self.layers) > idx + 2) and (self.layers[idx + 1].__name__() == 'Pooling') else 1
                x, self.recurrent_inputs[idx], self.topdown_inputs[idx] = \
                    layer([x, self.recurrent_outputs[idx], self.topdown_outputs[idx + next_layer]],
                          feedforward_flag=feedforward_flag, recurrent_flag=recurrent_flag, topdown_flag=topdown_flag)
                self.recurrent_outputs[idx] = x
                self.topdown_outputs[idx] = x
            t += 1
            output = x

        return output, self.recurrent_inputs, self.topdown_inputs

    def cuda(self, device):
        super(MultiDirectional, self).cuda(device)
        for idx in range(len(self.recurrent_inputs)):
            self.recurrent_inputs[idx] = self.recurrent_inputs[idx].cuda(device)

        for idx in range(len(self.topdown_inputs)):
            self.topdown_inputs[idx] = self.topdown_inputs[idx].cuda(device)

        for idx in range(len(self.recurrent_outputs)):
            self.recurrent_outputs[idx] = self.recurrent_outputs[idx].cuda(device)

        for idx in range(len(self.topdown_outputs)):
            self.topdown_outputs[idx] = self.topdown_outputs[idx].cuda(device)
