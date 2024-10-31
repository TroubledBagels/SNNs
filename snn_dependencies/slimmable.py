import json
import os
import torch.nn as nn
import torch

class SwitchableBatchNorm2d(nn.Module): # has multiple batch normalisation layers corresponding to network width
    def __init__(self, num_features_list, model_name):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list # sets the number of features in the list
        self.num_features = max(num_features_list) # finds the maximum number of features
        self.params = json.load(open(os.path.join('snn_models/' + model_name + '.json'), 'r')) # loads the parameters from the model file
        print("SBatch", num_features_list)

        bns = [] # creates the list of batch normalisation layers
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i)) # fills list with batch normalisation layers for each number of features
        self.bn = nn.ModuleList(bns) # stored in a module list, allowing dynamic switching in forward pass

        self.width_mult = max(self.params['width_mult_list']) # sets the default width multiplier to the maximum width multiplier in the list
        self.ignore_model_profiling = True # sets the model profiling to ignore - e.g. for measuring FLOPs

    def forward(self, input):
        idx = self.params['width_mult_list'].index(self.width_mult) # finds the correct index
        if self.width_mult == 0.5:
            print(input.size())
            print("SBatch", self.num_features_list)
        # if self.width_mult == 0.25:
        #     print(self.num_features_list)

        expected_features = self.num_features_list[idx]
        input_num_channels = input.size(1)

        if input_num_channels != expected_features:
            raise ValueError(f"Expected {expected_features} input channels, got {input_num_channels}")

        return self.bn[idx](input) # returns the batch normalisation layer at the index

    def print(self):
        print(self.bn)


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, model_name, in_channels_list, out_channels_list, kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True, starting_layer=False):
        super(SlimmableConv2d, self).__init__(max(in_channels_list), max(out_channels_list), kernel_size, stride=stride,
                                              padding=padding, dilation=dilation, groups=max(groups_list), bias=bias)
        # initialise the layer with the max input and output channels, kernel size, stride, padding, dilation, groups, and bias

        self.params = json.load(open(os.path.join('snn_models/' + model_name + '.json'), 'r')) # loads the parameters from the model file
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        print("Conv2d", out_channels_list)
        self.groups_list = groups_list # sets the groups list to the groups list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))] # ensures that groups_list has same length as in_channels_list

        self.starting_layer = starting_layer
        self.finished_first_train = False
        self.fully_trained = False

        self.weights = []

        self.width_mult = max(self.params['width_mult_list']) # sets the default width multiplier to the maximum width multiplier in the list

    def forward(self, inp):
        idx = self.params['width_mult_list'].index(self.width_mult) # sets the index to the index of the current width multiplier in the list

        self.in_channels = self.in_channels_list[idx] # dynamically set the input and output channels
        self.out_channels = self.out_channels_list[idx]
        out_channels_bottom = 0
        in_channels_bottom = 0
        self.groups = self.groups_list[idx]

        if self.width_mult == 0.5:
            print("Conv2d", self.out_channels)

        weight = self.load_weights()
        # if not self.starting_layer:
        #     if idx > 0:
        #         weight = self.weights[0]
        #         out_channels_bottom = self.out_channels_list[idx - 1]
        #         in_channels_bottom = self.in_channels_list[idx - 1]
        #         for i in range(1, idx):
        #             torch.cat((weight, self.weights[i]), 1)
        #         if idx == 1 and 9 in self.in_channels_list: in_channels_bottom = 0
        #         print("out_channels:", out_channels_bottom, self.out_channels)
        #         print("in_channels:", in_channels_bottom, self.in_channels)
        #         print(weight.size())
        #         print(self.weight[out_channels_bottom:self.out_channels, in_channels_bottom:self.in_channels, :, :].size())
        #         weight = torch.cat((weight, self.weight[out_channels_bottom:self.out_channels, in_channels_bottom:self.in_channels, :, :]), 1)
        #         print(weight.size())
        #     else:
        #         weight = self.weight[out_channels_bottom:self.out_channels, in_channels_bottom:self.in_channels, :, :] # slices the weight tensor to only those used
        # else:
        #     weight = self.weight[:self.out_channels, :self.in_channels, :, :]

        if idx > 0:
            if len(self.weights) != len(self.out_channels_list):
                weight = torch.cat((weight, self.weight[:self.out_channels_list[idx-1], self.in_channels_list[idx-1]:self.in_channels_list[idx], :, :]), 1)
                weight = torch.cat((weight, self.weight[self.out_channels_list[idx-1]:self.out_channels_list[idx], :self.in_channels_list[idx], :, :]), 0)
        else:
            weight = self.weight[:self.out_channels, :self.in_channels, :, :]

        if self.fully_trained:
            weight = self.load_weights()

        if self.bias is not None:
            bias = self.bias[:self.out_channels] # slices the bias if that is required
        else:
            bias = self.bias

        y = nn.functional.conv2d(inp, weight, bias, self.stride, self.padding, self.dilation, self.groups) # applies the convolution operation
        # print(weight[0][0][0])
        # print(weight.size())
        # if self.width_mult == 0.25:
        #     print("y:", y.size())
        return y # returns the convolutional layer having processed the input

    def load_weights(self):
        weight = None
        if self.finished_first_train:
            weight = self.weights[0]
            for x, y in self.weights[1:]:
                weight = torch.cat((weight, x), 1)
                weight = torch.cat((weight, y), 0)
        return weight

    def save_weights(self, idx):
        if idx == 0:
            self.weights.append(self.weight[:self.out_channels_list[idx], :self.in_channels_list[idx], :, :])
        else:
            x = self.weight[:self.out_channels_list[idx-1], self.in_channels_list[idx-1]:self.in_channels_list[idx], :, :]
            y = self.weight[self.out_channels_list[idx-1]:self.out_channels_list[idx], :self.in_channels_list[idx], :, :]
            self.weights.append((x, y))
        pass

    def end_training(self, wm):
        idx = self.params['width_mult_list'].index(wm) # sets the index to the index of the current width multiplier in the list
        self.save_weights(idx) # saves the weights at the index
        self.finished_first_train = True
        if len(self.weights) == len(self.out_channels_list):
            self.fully_trained = True

    def print(self):
        print(self.in_channels)
        print(self.out_channels)
        print(self.bias)
        print(self.stride)


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, model_name, bias=True):
        super(SlimmableLinear, self).__init__(max(in_features_list), max(out_features_list), bias=bias) # initialise the model with maximum channel values
        self.in_features_list = in_features_list # sets the input features list
        self.out_features_list = out_features_list # sets the output features list
        self.params = json.load(open(os.path.join('snn_models/' + model_name + '.json'), 'r')) # loads the parameters from the model file
        self.width_mult = max(self.params['width_mult_list']) # sets the default width multiplier to the maximum width multiplier in the list
        self.weights = []

    def forward(self, input):
        idx = self.params['width_mult_list'].index(self.width_mult) # sets the index to the index of the current width multiplier in the list
        self.in_features = self.in_features_list[idx] # sets the input features to the input features at the index
        self.out_features = self.out_features_list[idx] # sets the output features to the output features at the index
        weight = self.weight[:self.out_features, :self.in_features] # slices the weight to the weight of the current output and input features
        if self.bias is not None:
            bias = self.bias[:self.out_features] # slices the bias to the bias of the current output features
        else:
            bias = self.bias
        y = nn.functional.linear(input, weight, bias) # applies the linear operation
        return y

    def end_training(self, wm):
        idx = self.params['width_mult_list'].index(wm)
        in_features_top = self.in_features_list[idx]
        out_features_top = self.out_features_list[idx]
        in_features_bottom = 0
        out_features_bottom = 0
        if idx > 0:
            in_features_bottom = self.in_features_list[idx - 1]
            out_features_bottom = self.out_features_list[idx - 1]
        self.weights.append(self.weight[out_features_bottom:out_features_top, in_features_bottom:in_features_top])

    def print(self):
        print(self.in_features)
        print(self.out_features)
        print(self.bias)


def make_divisible(v, divisor=8, min_value=1): # makes a given value, v, divisible by a divisor
    if min_value is None:
        min_value = divisor # if the minimum value is not set, set it to the divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # the above rounds v to the nearest possible multiple of divisor
    # v + divisor / 2 ensures rounding rather than truncation
    # int(...) // divisor * divisor divides the adjusted b ny the divisor, truncating the result, and multiplies back by the divisor to ensure it is a multiple
    # max(min_value, ...) ensures that the new value is at least the minimum value
    if new_v < 0.9 * v:
        new_v += divisor # ensures the rounding doesn't decrease more than 10% of the original value, otherwise adds the divisor
    return new_v # return the new value

# def bn_calibration_init(m): # resets the model for calibration with batch normalisation
#     if getattr(m, 'track_running_stats', False):
#         m.reset_running_stats() # reset the running stats
#         m.training = True
#         if getattr(FLAGS, 'cumulative_bn_stats', False):
#             m.momentum = None
