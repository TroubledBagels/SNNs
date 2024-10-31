import json, os, torch
import time

import torch.nn as nn

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, model_name, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.nfl = num_features_list
        self.model_name = model_name
        self.params = json.load(open("snn_models/" + model_name + ".json"))
        self.width_mult = max(self.params['width_mult_list'])

        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)

    def forward(self, x):
        idx = self.params['width_mult_list'].index(self.width_mult) + 1

        return self.bn[idx](x)

class SlimmableConv2d(nn.Conv2d):
    def __init__(self, model_name, icl, ocl, kernel_size, stride=1, padding=0, dilation=1, groups_list=[1], bias=True,
                 starting_layer=False):
        super(SlimmableConv2d, self).__init__(max(icl), max(ocl), kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, groups=max(groups_list), bias=bias)
        self.model_name = model_name
        self.icl = icl # in_channels_list assignment
        self.ocl = ocl # out_channels_list assignment
        self.starting_layer = starting_layer
        self.finished_first_training = False
        self.finished_final_training = False

        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(self.icl))]

        self.params = json.load(open("snn_models/" + model_name + ".json"))
        self.width_mult = max(self.params['width_mult_list'])

        self.weights = []

    def forward(self, inp):
        idx = self.params['width_mult_list'].index(self.width_mult) + 1

        if self.starting_layer:
            inp = inp.repeat(1, idx, 1, 1)

        self.groups = self.groups_list[idx]

        weight = self.load_weights()

        y = nn.functional.conv2d(inp, weight, None, self.stride, self.padding, self.dilation, self.groups)

        return y

    def load_weights(self): # see accompanying image in README.md for data structure example
        idx = self.params['width_mult_list'].index(self.width_mult) + 1

        weight = None
        if self.finished_first_training:
            weight = self.weights[0]

            for x, y in self.weights[1:(min(idx, len(self.weights)))]:
                # print("weight:", weight.size())
                # print("x:", x.size())
                weight = torch.cat((weight, x), 1)
                weight = torch.cat((weight, y), 0)

            if not self.finished_final_training:
                x = self.weight[:self.ocl[idx-1], self.icl[idx-1]:self.icl[idx], :, :]
                y = self.weight[self.ocl[idx-1]:self.ocl[idx], :self.icl[idx], :, :]
                # print("weight size:", weight[:self.ocl[idx-1], self.icl[idx-1]:self.icl[idx], :, :].size())
                # print("x size:", x.size())
                # print("y size:", y.size())
                weight = torch.cat((weight, x), 1)
                weight = torch.cat((weight, y), 0)

        else:
            weight = self.weight[:self.ocl[idx], :self.icl[idx], :, :]

        return weight

    def save_weights(self):
        idx = self.params['width_mult_list'].index(self.width_mult) + 1

        if not self.finished_first_training:
            self.weights.append(self.weight[:self.ocl[idx], :self.icl[idx], :, :].clone())
            return

        if not self.finished_final_training:
            x = self.weight[:self.ocl[idx-1], self.icl[idx-1]:self.icl[idx], :, :].clone()
            y = self.weight[self.ocl[idx-1]:self.ocl[idx], :self.icl[idx], :, :].clone()
            self.weights.append((x, y))

    def end_training(self):
        self.save_weights()
        self.finished_first_training = True
        if len(self.weights) == len(self.ocl)-1:
            self.finished_final_training = True


class SlimmableLinear(nn.Linear):
    def __init__(self, model_name, ifl, ofl, bias=True):
        super(SlimmableLinear, self).__init__(max(ifl), max(ofl), bias=bias)
        self.ifl = ifl # in_features_list assignment
        self.ofl = ofl # out_features_list assignment

        self.params = json.load(open("snn_models/" + model_name + ".json"))
        self.width_mult = max(self.params['width_mult_list'])

        self.finished_first_training = False
        self.finished_final_training = False

        self.weights = []

    def forward(self, inp):
        idx = self.params['width_mult_list'].index(self.width_mult) + 1

        self.in_features = self.ifl[idx]
        self.out_features = self.ofl[idx]

        weight = self.load_weights()

        bias = self.bias[:self.ofl[idx]] if self.bias is not None else None

        y = nn.functional.linear(inp, weight, bias)

        return y

    def load_weights(self):
        idx = self.params['width_mult_list'].index(self.width_mult) + 1

        weight = None
        if not self.finished_first_training:
            weight = self.weight[:self.ofl[idx], :self.ifl[idx]]
        else:
            weight = self.weights[0]

            for x, y in self.weights[1:(min(idx, len(self.weights)))]:
                weight = torch.cat((weight, x), 1)
                weight = torch.cat((weight, y), 0)

            if not self.finished_final_training:
                x = self.weight[:self.ofl[idx - 1], self.ifl[idx - 1]:self.ifl[idx]]
                y = self.weight[self.ofl[idx - 1]:self.ofl[idx], :self.ifl[idx]]
                weight = torch.cat((weight, x), 1)
                weight = torch.cat((weight, y), 0)

        return weight

    def end_training(self):
        self.save_weights()
        self.finished_first_training = True
        if len(self.weights) == len(self.ofl)-1:
            self.finished_final_training = True

    def save_weights(self):
        idx = self.params['width_mult_list'].index(self.width_mult) + 1

        if not self.finished_first_training:
            self.weights.append(self.weight[:self.ofl[idx], :self.ifl[idx]].clone())
            return

        if not self.finished_final_training:
            x = self.weight[:self.ofl[idx-1], self.ifl[idx - 1]:self.ifl[idx]].clone()
            y = self.weight[self.ofl[idx - 1]:self.ofl[idx], :self.ifl[idx]].clone()
            self.weights.append((x, y))
