import torch
import torch.nn as nn


class Pooling(nn.Module):
    def __init__(self, parameters):
        """This is a unit that contains triple convolutional/conv_transpose layers to implement feedforward, recurrent,
         and top-down processing of images"""

        super(Pooling, self).__init__()

        self.parameters = parameters
        self.layer = nn.MaxPool2d(parameters['kernel_size'],
                                  stride=parameters['stride'],
                                  padding=parameters['padding'])

    def forward(self, input_data, feedforward_flag=True, recurrent_flag=True, topdown_flag=True, data_size=None):
        """This supports any of the combinations between feedforward, recurrent, top_down by turning on an off
        any of the flags for that pathway. It can even go in a backward direction if the feedforward flag is turned
        off and data_size is given. Input data should be a list of three inputs in full mode where feedforward input
        is the first element, recurrent is second and top-down is third. In case of absence of feedforward with any of
        the later inputs presence, any placeholder should be inserted in its place. Same goes for absence of recurrent
        input and presence of top-down."""

        x = input_data[0]
        output = self.layer(x)

        return output, None, None

    def output_size(self, input_size):
        """Assuming square kernels"""
        width_height = ((input_size[1] +
                         2 * self.parameters['padding']
                         - self.parameters['kernel_size'])
                        / self.parameters['stride']) + 1
        width_height = int(width_height)
        return [input_size[0], width_height, width_height]

    @staticmethod
    def __name__():
        return 'Pooling'
