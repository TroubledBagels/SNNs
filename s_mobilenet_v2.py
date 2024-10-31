import math
import torch.nn as nn

from snn_dependencies.slimmable import SwitchableBatchNorm2d, SlimmableConv2d
from snn_dependencies.slimmable import make_divisible
from utils.config import FLAGS

class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []

        expand_inp = [1 * expand_ratio for i in inp] # sets the expanded input to the input multiplied by the expand ratio
        if expand_ratio != 1:
            layers += [SlimmableConv2d(inp, expand_inp, 1, 1, 0, bias=False),
                       SwitchableBatchNorm2d(expand_inp), nn.ReLU6(inplace=True)]
            # if the expand ratio is not 1, append a convolutional layer, batch normalisation layer, and ReLU activation function to the list of layers
        layers += [SlimmableConv2d(expand_inp, expand_inp, kernel_size=3, stride=stride, padding=1,
                                   groups_list=expand_inp, bias=False),
                   SwitchableBatchNorm2d(expand_inp), nn.ReLU6(inplace=True),
                   SlimmableConv2d(expand_inp, outp, 1, 1, 0, bias=False),
                   SwitchableBatchNorm2d(outp)]
        # append a convolutional layer, batch normalisation layer, ReLU activation function, convolutional layer, and batch normalisation layer to the list of layers
        self.body = nn.Sequential(*layers) # sets the body to the layers

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        # t is the expansion factor
        # c is the number of output channels
        # n is the number of times to repeat the inverted residual blocks
        # s is the stride
        self.block_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.features = []
        assert input_size % 32 == 0
        channels = [make_divisible(32 * width_mult) for width_mult in FLAGS.width_mult_list] # sets the channels to the number of channels in the list

        # head
        self.outp = make_divisible(1280 * max(FLAGS.width_mult_list)) if max(FLAGS.width_mult_list) > 1.0 else 1280 # sets the output to the maximum width multiplier
        first_stride = 2
        self.features.append(
            nn.Sequential(
                SlimmableConv2d([3 for _ in range(len(channels))], channels, kernel_size=3, stride=first_stride,
                                padding=1, bias=False), # creates a convolutional layer
                SwitchableBatchNorm2d(channels),
                nn.ReLU6(inplace=True)
            )
        )

        # body
        for t, c, n, s in self.block_setting: # for each layer of the block setting
            outp = [make_divisible(c * width_mult) for width_mult in FLAGS.width_mult_list]
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(channels, outp, s, t)) # append the inverted residual block to the features
                else:
                    self.features.append(InvertedResidual(channels, outp, 1, t)) # append the inverted residual block to the features
                channels = outp

        #tail
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(channels, [self.outp for _ in range(len(channels))], kernel_size=1, stride=1, padding=0,
                                bias=False), # creates a convolutional layer
                nn.BatchNorm2d(self.outp), # creates a batch normalisation layer
                self.ReLU6(inplace=True)
            )
        )
        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(nn.Linear(self.outp, num_classes))
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.outp)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
