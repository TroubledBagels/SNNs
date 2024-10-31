import os
import sys
import torch
import time
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import json
import snn_dependencies.slimmable as slim
import snn_dependencies.slimmable_parse as slim_parse

'''
    Author: Ben Hatton (f08473bh), University of Manchester
    Date: 11/10/2024
    Description: The main driver behind the SNN model, using slimmable.py (taken from the Slimmable Networks paper)
'''

class SNN(nn.Module):
    def __init__(self, params):
        super(SNN, self).__init__()
        if params["verbosity"] > 0: print("[OK] Initialising SNN model...")
        self.layer_num = len(params['max_channels_per_layer_list'])
        self.layers = []
        self.width_mult_list = params['width_mult_list']
        self.verbosity = params['verbosity']
        stride = params['stride']
        kernel_size = params['kernel_size']
        self.width_mult = max(params['width_mult_list'])

        if params["verbosity"] > 1: print("[OK] Layer number: ", self.layer_num)
        if params["verbosity"] > 1: print("[OK] Layers: ", params['max_channels_per_layer_list'])
        icl = [9 for i in range(len(params['width_mult_list']))]
        ocl = []
        for i in range(len(params['width_mult_list'])):
            ocl.append(int(params['max_channels_per_layer_list'][0] * params['width_mult_list'][i]))

        self.layers.append(slim.SlimmableConv2d(params['model_name'], icl, ocl, kernel_size, stride, 1, bias=False, starting_layer=True))
        self.layers.append(slim.SwitchableBatchNorm2d(ocl, params['model_name']))
        self.layers.append(nn.ReLU6(inplace=True))

        out_channels_list = []

        for i in range (1, self.layer_num):
            cur_in = params['max_channels_per_layer_list'][i-1]
            cur_out = params['max_channels_per_layer_list'][i]
            in_channels_list = []
            out_channels_list = []
            for j in range(len(params['width_mult_list'])):
                in_channels_list.append(int(cur_in * params['width_mult_list'][j]))
                out_channels_list.append(int(cur_out * params['width_mult_list'][j]))
            self.layers.append(slim.SlimmableConv2d(params['model_name'], in_channels_list, out_channels_list, kernel_size, stride, 1, bias=False))
            self.layers.append(slim.SwitchableBatchNorm2d(out_channels_list, params['model_name']))
            self.layers.append(nn.ReLU6(inplace=True))

        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layers.append(nn.Flatten())

        intermediary_channels = [params['output_size'] ** 2 for width in self.width_mult_list]
        self.layers.append(slim.SlimmableLinear(out_channels_list, intermediary_channels, params['model_name']))
        self.layers.append(nn.ReLU6(inplace=True))

        self.layers.append(slim.SlimmableLinear(intermediary_channels, [params['output_size'] for _ in self.width_mult_list], params['model_name']))
        self.layers.append(nn.Softmax(dim=1))

        self.layers = nn.ModuleList(self.layers)

        if params["verbosity"] > 0: print("[OK] SNN model initialised!")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def print(self):
        for layer in self.layers:
            if isinstance(layer, slim.SlimmableConv2d) or isinstance(layer, slim.SlimmableLinear) or isinstance(layer, slim.SwitchableBatchNorm2d):
                layer.print()
            else:
                print("Activation or pooling layer")

    def change_width_mult(self, wm):
        self.width_mult = wm
        for layer in self.layers:
            if isinstance(layer, slim.SlimmableConv2d) or isinstance(layer, slim.SlimmableLinear) or isinstance(layer, slim.SwitchableBatchNorm2d):
                layer.width_mult = wm

    def end_training(self):
        for layer in self.layers:
            if isinstance(layer, slim.SlimmableConv2d) or isinstance(layer, slim.SlimmableLinear):
                layer.end_training(self.width_mult)

def forward_loss(model, criterion, input, target, verbosity):
    output = model(input)
    loss = torch.mean(criterion(output, target))
    _, pred = output.topk(5)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    correct_k = []
    for k in [1, 5]:
        correct_k.append(correct[:k].float().sum(0))
    tensor = torch.cat([loss.view(1)] + correct_k, dim=0)

    return tensor, output

def train(dataloader, model, criterion, optimiser, verbosity):
    size = len(dataloader.dataset)
    model.train()
    loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.cuda(), y.cuda()
        optimiser.zero_grad()
        loss_list = []

        if model.width_mult == 1.0:
            loss, soft_target = forward_loss(model, criterion, X, y, verbosity)
        else:
            loss, _ = forward_loss(model, criterion, X, y, verbosity)
        pred = model(X)
        loss = criterion(pred, y)
        loss_list.append(loss)

        loss.backward()
        optimiser.step()

        if verbosity > 1 and batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"    loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss

def test(dataloader, model, criterion, verbosity):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    loss_list = []
    correct_list = []
    times = []
    for wm in sorted(model.width_mult_list):
        model.change_width_mult(wm)
        start_time = time.time()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.cuda(), y.cuda()
                if wm == 1.0:
                    loss, soft_target = forward_loss(model, criterion, X, y, verbosity)
                else:
                    loss, _ = forward_loss(model, criterion, X, y, verbosity)
                pred = model(X)
                test_loss += criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss_list.append(test_loss)
            correct_list.append(correct)
            test_loss, correct = 0, 0
            times.append(time.time() - start_time)
    loss_list = [x / num_batches for x in loss_list]
    correct_list = [x / size for x in correct_list]
    if verbosity > 0:
        for i in range(len(model.width_mult_list)):
            print(f"    Width: {model.width_mult_list[i]}, Accuracy: {(100*correct_list[i]):>0.1f}%, Avg loss: {loss_list[i]:>8f}, Latency: {times[i]:>0.3f} \n")
    return test_loss, correct

def test_single(dataloader, model, criterion, verbosity):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    start_time = time.time()
    runtime = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            if model.width_mult == 1.0:
                loss, soft_target = forward_loss(model, criterion, X, y, verbosity)
            else:
                loss, _ = forward_loss(model, criterion, X, y, verbosity)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        runtime = time.time() - start_time
    test_loss /= num_batches
    correct /= size
    if verbosity > 0:
        print(f"    Width: {model.width_mult:>0.3f}, Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, Latency: {runtime:>0.3f} \n")
    return test_loss, correct

def run_test(dataloader, model, criterion, verbosity, w):
    model.eval()
    loss, correct = 0, 0
    for X, y in dataloader:
        loss_temp, correct_temp = test_one(X, y, model, criterion, verbosity, w)
        loss += loss_temp
        correct += correct_temp
    loss /= len(dataloader)
    correct /= len(dataloader)
    if verbosity > 0:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")


def test_one(X, y, model, criterion, verbosity, wmlist):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        X, y = X.cuda(), y.cuda()
        model.change_width_mult(wmlist[0])
        batch_confidence = 0
        while batch_confidence < 0.9:
            confidences = []
            pred = model(X)
            for i, x in enumerate(pred):
                confidences.append(x[y[i]])
            batch_confidence = sum(confidences) / len(confidences)
            if batch_confidence < 0.9:
                if wmlist.index(model.width_mult) == len(wmlist) - 1:
                    if verbosity > 0:
                        print("[WARNING] Model cannot reach 90% confidence!")
                    break
                else:
                    model.change_width_mult(wmlist[wmlist.index(model.width_mult) + 1])
                    if verbosity > 1:
                        print("[OK] Switching width multiplier to:", model.width_mult)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss += criterion(pred, y).item()
    test_loss /= len(X)
    correct /= len(X)
    if verbosity > 0:
        print(f"    Width: {model.width_mult:>0.3f}, Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

def load_data(dataset, batch_size, verbosity):
    if verbosity > 1: print("[OK] Attempting load of " + dataset + "...")
    if dataset == "CIFAR10":
        transform = Compose([ToTensor(), Lambda(lambda x: x.repeat(3, 1, 1))])
        train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    elif dataset == "CIFAR100":
        transform = Compose([ToTensor(), Lambda(lambda x: x.repeat(3, 1, 1))])
        train_data = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    elif dataset == "MNIST":
        transform = Compose([ToTensor()])
        train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    else:
        print("[ERROR] Invalid dataset given: ", dataset)
        sys.exit(1)
    if verbosity > 1: print("[OK] " + dataset + " loaded!")

    if verbosity > 1: print("[OK] Creating data loaders...")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    if verbosity > 1: print("[OK] Data loaders created!")

    return train_loader, test_loader


def main(params):
    model = SNN(params).cuda()
    if params['verbosity'] > 2:
        print("[OK] Printing model..."
              "-------------------------------")
        model.print()
        print("-------------------------------")

    print(model.parameters())

    if params['verbosity'] > 0: print("[OK] Loading loss function and optimiser...")
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    if params['verbosity'] > 0: print("[OK] Loss function and optimiser loaded!")

    if params['verbosity'] > 0: print("[OK] Loading data...")
    train_dl, test_dl = load_data(params['dataset'], params['batch_size'], params['verbosity'])
    if params['verbosity'] > 0: print("[OK] Data loaded!")

    if params['verbosity'] > 0: print("[OK] Training model...")
    for i in range(len(model.width_mult_list)):
        model.change_width_mult(model.width_mult_list[i])
        for t in range(params['epochs']):
            if params['verbosity'] > 0: print(f"[OK] ------------------------------- Mult: {model.width_mult_list[i]} | Epoch: {t+1} -------------------------------")
            train_loss = train(train_dl, model, criterion, optimiser, params['verbosity'])
            test_loss, accuracy = test_single(test_dl, model, criterion, params['verbosity'])
        if params['verbosity'] > 0: print("[OK] Switching mult...")
        model.end_training()
    if params['verbosity'] > 0: print("[OK] Model trained!")

    if params['verbosity'] > 0: print("[OK] Testing model...")
    run_test(test_dl, model, criterion, params['verbosity'], params['width_mult_list'])
    if params['verbosity'] > 0: print("[OK] Model tested!")

if __name__ == "__main__":
    params = slim_parse.parse_params(sys.argv)
    main(params)