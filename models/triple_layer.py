import torch
import torch.nn as nn


class TripleLayerConv(nn.Module):
    def __init__(self, input_channels, output_channels, topdown_channels, feedforward_recurrent_parameters,
                 topdown_parameters, dropout=None):
        """This is a unit that contains triple convolutional/conv_transpose layers to implement feedforward, recurrent,
         and top-down processing of images"""

        super(TripleLayerConv, self).__init__()

        # save some parameters for later
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.feedforward_parameters = feedforward_recurrent_parameters
        self.dropout = dropout

        # Model parameters based on inputs
        activation_function = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }
        self.feedforward_activation = activation_function[feedforward_recurrent_parameters['activation']]
        self.recurrent_activation = activation_function[feedforward_recurrent_parameters['activation']]
        self.topdown_activation = activation_function[topdown_parameters['activation']]

        self.feedforward_layer = nn.Conv2d(input_channels, output_channels,
                                           kernel_size=feedforward_recurrent_parameters['kernel_size'],
                                           stride=feedforward_recurrent_parameters['stride'],
                                           padding=feedforward_recurrent_parameters['padding'],
                                           dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.recurrent_layer = nn.ConvTranspose2d(output_channels, input_channels,
                                                  kernel_size=feedforward_recurrent_parameters['kernel_size'],
                                                  stride=feedforward_recurrent_parameters['stride'],
                                                  padding=feedforward_recurrent_parameters['padding'],
                                                  dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.topdown_layer = nn.ConvTranspose2d(topdown_channels, input_channels,
                                                kernel_size=topdown_parameters['kernel_size'],
                                                stride=topdown_parameters['stride'],
                                                padding=topdown_parameters['padding'],
                                                dilation=1, groups=1, bias=True, padding_mode='zeros')

    def forward(self, input_data, feedforward_flag=True, recurrent_flag=True, topdown_flag=True, data_size=None):
        """This supports any of the combinations between feedforward, recurrent, top_down by turning on an off
        any of the flags for that pathway. It can even go in a backward direction if the feedforward flag is turned
        off and data_size is given. Input data should be a list of three inputs in full mode where feedforward input
        is the first element, recurrent is second and top-down is third. In case of absence of feedforward with any of
        the later inputs presence, any placeholder should be inserted in its place. Same goes for absence of recurrent
        input and presence of top-down."""

        if feedforward_flag:
            x_feedforward = input_data[0]
        else:
            x_feedforward = torch.zeros(data_size)

        if recurrent_flag:
            x_recurrent = input_data[1]
            x_recurrent = self.recurrent_activation(self.recurrent_layer(x_recurrent))
        else:
            x_recurrent = torch.zeros(x_feedforward.shape)

        if topdown_flag:
            x_topdown = input_data[2]
            if len(x_topdown.shape) < 3:
                x_topdown = x_topdown.contiguous().view(-1, x_topdown.shape[-1], 1, 1)
            x_topdown = self.topdown_activation(self.topdown_layer(x_topdown))
        else:
            x_topdown = torch.zeros(x_feedforward.shape)

        x = x_feedforward + x_recurrent + x_topdown

        output = self.feedforward_activation(self.feedforward_layer(x))
        if self.dropout is not None:
            output = nn.Dropout(0.5)(output)

        return output, x_recurrent, x_topdown

    def output_size(self, input_size):
        """Assuming square kernels"""
        assert input_size[0] == self.input_channels
        width_height = ((input_size[1] +
                         2 * self.feedforward_parameters['padding']
                         - self.feedforward_parameters['kernel_size'])
                        / self.feedforward_parameters['stride']) + 1
        width_height = int(width_height)
        return [self.output_channels, width_height, width_height]

    @staticmethod
    def __name__():
        return 'TripleLayerConv'


class TripleLayerFc(nn.Module):
    def __init__(self, input_channels, output_channels, topdown_channels, feedforward_recurrent_parameters,
                 topdown_parameters, dropout=None):
        """This is a unit that contains triple fully-connected layers to implement feedforward, recurrent,
         and top-down processing of 1-D data"""

        super(TripleLayerFc, self).__init__()

        # save some parameters for later
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dropout = dropout

        # Model parameters based on inputs
        activation_function = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=1),
        }
        self.feedforward_activation = activation_function[feedforward_recurrent_parameters['activation']]
        self.recurrent_activation = activation_function[feedforward_recurrent_parameters['activation']]
        self.topdown_activation = activation_function[topdown_parameters['activation']]

        self.feedforward_layer = nn.Linear(input_channels, output_channels)
        self.recurrent_layer = nn.Linear(output_channels, input_channels)
        self.topdown_layer = nn.Linear(topdown_channels, input_channels)

    def forward(self, input_data, feedforward_flag=True, recurrent_flag=True, topdown_flag=True, data_size=None):
        """This supports any of the combinations between feedforward, recurrent, top_down by turning on an off
        any of the flags for that pathway. It can even go in a backward direction if the feedforward flag is turned
        off and data_size is given. Input data should be a list of three inputs in full mode where feedforward input
        is the first element, recurrent is second and top-down is third. In case of absence of feedforward with any of
        the later inputs presence, any placeholder should be inserted in its place. Same goes for absence of recurrent
        input and presence of top-down."""

        if feedforward_flag:
            x_feedforward = input_data[0]
        else:
            x_feedforward = torch.zeros(data_size)

        if len(x_feedforward.shape) > 2:
            x_feedforward = x_feedforward.contiguous().view(-1, self.input_channels)

        if recurrent_flag:
            x_recurrent = input_data[1]
            x_recurrent = self.recurrent_activation(self.recurrent_layer(x_recurrent))
        else:
            x_recurrent = torch.zeros(x_feedforward.shape)

        if topdown_flag:
            x_topdown = input_data[2]
            x_topdown = self.topdown_activation(self.topdown_layer(x_topdown))
        else:
            x_topdown = torch.zeros(x_feedforward.shape)

        x = x_feedforward + x_recurrent + x_topdown

        output = self.feedforward_activation(self.feedforward_layer(x))
        if self.dropout is not None:
            output = nn.Dropout(0.5)(output)

        return output, x_recurrent, x_topdown

    def output_size(self, input_size):
        """Assuming square kernels"""
        assert input_size[0] == self.input_channels
        return [self.output_channels]

    @staticmethod
    def __name__():
        return 'TripleLayerFc'
