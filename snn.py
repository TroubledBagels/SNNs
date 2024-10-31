import json
import sys
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import snn_dependencies.slimmable_v2 as slim
import snn_dependencies.slimmable_parse as sp

'''
    Author: Ben Hatton (f08473bh), University of Manchester
    DateL 11/10/2024
    Description: This is the main file for the slimmbale neural network (SNN) project
'''


class SNN(nn.Module):
    def __init__(self, params):
        super(SNN, self).__init__()

        if params['verbosity'] > 0: print("[OK] Initialising SNN model...")

        # Initialise the model parameters
        self.layer_num = len(params['max_channels_per_layer_list'])
        self.width_mult_list = params['width_mult_list']
        self.verbosity = params['verbosity']
        self.name = params['model_name']
        self.params = params
        stride = params['stride']
        kernel_size = params['kernel_size']

        self.layers = []

        if params['verbosity'] > 1:
            print("[OK] Layer number:", self.layer_num)
            print("[OK] Layers:", self.params['max_channels_per_layer_list'])

        # Create the first layer
        icl = [0]
        for i in range(len(self.width_mult_list)):
            icl.append(3 * (i+1))
        ocl = [0]
        for i in range(len(params['width_mult_list'])):
            ocl.append(int(params['max_channels_per_layer_list'][0] * params['width_mult_list'][i]))

        self.layers.append(slim.SlimmableConv2d(self.name, icl, ocl, kernel_size, stride=stride, starting_layer=True))
        self.layers.append(slim.SwitchableBatchNorm2d(self.name, ocl))
        self.layers.append(nn.ReLU6(inplace=True))

        # Create convolutional layers
        for i in range(1, self.layer_num):
            cur_in = int(params['max_channels_per_layer_list'][i-1])
            cur_out = int(params['max_channels_per_layer_list'][i])

            icl = [0]
            ocl = [0]
            for j in range(len(params['width_mult_list'])):
                icl.append(int(cur_in * params['width_mult_list'][j]))
                ocl.append(int(cur_out * params['width_mult_list'][j]))

            self.layers.append(slim.SlimmableConv2d(self.name, icl, ocl, kernel_size, stride=stride))
            self.layers.append(slim.SwitchableBatchNorm2d(self.name, ocl))
            self.layers.append(nn.ReLU6(inplace=True))

        # Create the linear layers
        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layers.append(nn.Flatten())

        ic = [0]
        for i in range(len(self.width_mult_list)):
            ic.append(int((params['output_size'] ** 3) * self.width_mult_list[i]))
        print(ic)
        self.layers.append(slim.SlimmableLinear(self.name, ocl, ic))
        self.layers.append(nn.ReLU6(inplace=True))

        ic2 = [0]
        for i in range(len(self.width_mult_list)):
            ic2.append(int((params['output_size'] ** 2) * self.width_mult_list[i]))
        self.layers.append(slim.SlimmableLinear(self.name, ic, ic2))
        self.layers.append(nn.ReLU6(inplace=True))

        self.layers.append(slim.SlimmableLinear(self.name, ic2, [params['output_size'] for _ in range(len(self.width_mult_list) + 1)]))
        self.layers.append(nn.Softmax(dim=1))

        # Create the model
        self.model = nn.Sequential(*self.layers)

        if params['verbosity'] > 0: print("[OK] SNN model initialised.")

    def forward(self, x):
        return self.model(x)

    def change_width_mult(self, width_mult):
        for layer in self.layers:
            if isinstance(layer, slim.SlimmableConv2d) or isinstance(layer, slim.SlimmableLinear) or isinstance(layer, slim.SwitchableBatchNorm2d):
                layer.width_mult = width_mult

    def end_training(self):
        for layer in self.layers:
            if isinstance(layer, slim.SlimmableConv2d) or isinstance(layer, slim.SlimmableLinear):
                layer.end_training()

    def print(self):
        print("[INFO] SNN model:")
        for layer in self.layers:
            print("[INFO]   ", layer)

def run_one(model, criterion, input, target, verbosity):
    output = model(input)
    loss = torch.mean(criterion(output, target))
    confidence = torch.mean(torch.max(output, 1)[0])
    return loss, confidence

def test_single(model, dataloader, criterion, device, verbosity):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, confidence = 0, 0, 0
    start_time = time.time()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            confidence += torch.mean(torch.max(pred, 1)[0]).item()

        runtime = time.time() - start_time
    test_loss /= num_batches
    correct /= size
    confidence /= num_batches
    return test_loss, correct, confidence, runtime

def train(model, dataloader, criterion, optimiser, verbosity, device):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()

        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()

        optimiser.step()

        if verbosity > 1 and batch % 100 == 0:
            print("[INFO]   loss: {loss:>4f} [{current}/{size:>5d}]".format(loss=loss.item(), current=batch * len(X), size=size))

def test_levels(model, dataloader, criterion, device, verbosity):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    if verbosity > 0:
        print("[INFO] Testing model on", size, "samples, in", num_batches, "batches")

    test_loss, correct, confidence = 0, 0, 0

    loss_list = []
    correct_list = []
    confidence_list = []
    times = []

    model.eval()
    for wm in sorted(model.width_mult_list):
        if verbosity > 1:
            print("[INFO] Testing model width of " + str(wm) + "x")

        model.change_width_mult(wm)
        start_time = time.time()

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                test_loss += criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                confidence += torch.mean(torch.max(pred, 1)[0]).item()

            loss_list.append(test_loss)
            correct_list.append(correct)
            confidence_list.append(confidence)

            test_loss, correct, confidence = 0, 0, 0

            times.append(time.time() - start_time)

    loss_list = [x / num_batches for x in loss_list]
    correct_list = [x / size for x in correct_list]
    confidence_list = [x / num_batches for x in confidence_list]

    if verbosity > 0:
        print("[INFO] Testing complete")
        for i in range(len(model.width_mult_list)):
            print("[INFO]   width_mult:", model.width_mult_list[i], "loss:", loss_list[i], "accuracy:", correct_list[i], "confidence:", confidence_list[i], "time:", times[i])

    return loss_list, correct_list, confidence_list, times

def load_data(dataset, batch_size, verbosity):
    if verbosity > 1: print("[OK] Loading dataset: " + dataset + "...")

    if dataset == "CIFAR10":
        transform = Compose([ToTensor(), Lambda(lambda x: x.repeat(1, 1, 1))])
        train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    elif dataset == "MNIST":
        transform = Compose([ToTensor(), Lambda(lambda x: x.view(1, 28, 28))])
        train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    elif dataset == "CIFAR100":
        transform = Compose([ToTensor(), Lambda(lambda x: x.repeat(1, 1, 1))])
        train_data = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    else:
        print("[ERROR] Invalid dataset given:", dataset)
        sys.exit(1)

    if verbosity > 1: print("[OK] Dataset " + dataset + " loaded.")

    if verbosity > 1: print("[OK] Creating dataloaders...")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    if verbosity > 1: print("[OK] Dataloaders created.")

    return train_loader, test_loader

def make_serialisable(w):
    for i in range(len(w)):
        if isinstance(w[i], torch.Tensor):
            w[i] = w[i].tolist()
        elif isinstance(w[i], tuple):
            x = []
            for j in range(len(w[i])):
                x.append(w[i][j].tolist())
    return w

def save_model(model):
    file = open("snn_models/" + model.name + "_weights.json", "w")
    json_out = {}
    print(model.layers)
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], slim.SlimmableConv2d) or isinstance(model.layers[i], slim.SlimmableLinear):
            layer_weights = []
            for j in range(len(model.layers[i].weights)):
                if isinstance(model.layers[i].weights[j], tuple):
                    layer_weights.append([model.layers[i].weights[j][0].tolist(), model.layers[i].weights[j][1].tolist()])
                else:
                    layer_weights.append(model.layers[i].weights[j].tolist())
            json_out[i] = layer_weights

    json.dump(json_out, file)

def load_model(name):
    file = open("snn_models/" + name + "_weights.json", "r")
    data = json.load(file)
    model = SNN(data)
    return model

def main(params):
    verbosity = params['verbosity']

    if verbosity > 0: print("[OK] Loading data...")
    train_loader, test_loader = load_data(params['dataset'], params['batch_size'], verbosity)
    if verbosity > 0: print("[OK] Data loaded.")

    if verbosity > 0: print("[OK] Setting device...")
    device = params['device']
    if device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SNN(params).to(device)
    if verbosity > 2:
        print("[INFO] Model:")
        model.print()

    if verbosity > 0: print("[OK] Initialising criterion and optimiser...")
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    if verbosity > 0: print("[OK] Criterion and optimiser initialised.")

    if verbosity > 0: print("[OK] Training model...")
    for i in range(len(model.width_mult_list)):
        model.change_width_mult(model.width_mult_list[i])

        for t in range(params['epochs']):
            if verbosity > 0: print(f"[OK] ------------------------------- Mult: {model.width_mult_list[i]} | Epoch: {t+1} -------------------------------")
            train(model, train_loader, criterion, optimiser, verbosity, device)

            loss, correct, confidence, runtime = test_single(model, test_loader, criterion, device, verbosity)
            if verbosity > 0: print("[INFO] Results for width_mult:", model.width_mult_list[i])
            if verbosity > 0: print("[INFO]    Loss:", loss)
            if verbosity > 0: print("[INFO]    Accuracy:", correct)
            if verbosity > 0: print("[INFO]    Confidence:", confidence)
            if verbosity > 0: print("[INFO]    Time:", runtime)

        if verbosity > 0: print("[OK] Finished training layer, changing width_mult...")
        model.end_training()

    if verbosity > 0: print("[OK] Model trained.")

    if verbosity > 0: print("[OK] Saving model...")
    save_model(model)
    if verbosity > 0: print("[OK] Model saved.")

    if verbosity > 0: print("[OK] Testing model layers...")
    loss_list, correct_list, confidence_list, times = test_levels(model, test_loader, criterion, device, verbosity)
    if verbosity > 0: print("[OK] Model tested.")
    print("[INFO] Results:")
    for i in range(len(model.width_mult_list)):
        print("[INFO]    Width: {wm}".format(wm=model.width_mult_list[i]))
        print("[INFO]        Loss: {loss}".format(loss=loss_list[i]))
        print("[INFO]        Accuracy: {acc}".format(acc=correct_list[i]))
        print("[INFO]        Confidence: {conf}".format(conf=confidence_list[i]))
        print("[INFO]        Time: {time}".format(time=times[i]))


if __name__ == "__main__":
    params = sp.parse_params(sys.argv)
    main(params)
