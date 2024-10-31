import os
import sys
import torch
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layers, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential()
        self.stack.add_module(f'linear{0}', nn.Linear(input_size, layers[0]))
        for i in range(1, len(layers)):
            self.add_module(f'linear{i}', nn.Linear(layers[i-1], layers[i]))
            self.add_module(f'relu{i}', nn.ReLU())
        self.stack.add_module(f'linear{len(layers)}', nn.Linear(layers[-1], output_size))
        self.stack.add_module(f'relu{len(layers)}', nn.ReLU())

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits


def train(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.cuda(), y.cuda()

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"    loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"    Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def plot_results(accuracy_results, loss_results):
    fig, ax = plt.subplots()
    ax.plot(accuracy_results, label='Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy Results')
    fig2, ax2 = plt.subplots()
    ax2.plot(loss_results, label='Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss Results')
    plt.ion()
    plt.pause(0.001)


def generate_image(img, actual, predicted):
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img.squeeze()
    img = Image.fromarray(np.uint8(img * 255), 'L')
    img = img.resize((128, 128))
    return img


def check_64(dataloader, model, params):
    images = []

    classes = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
    ]
    predicted = []
    actual = []

    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        for i in range(64):
            X, y = batch[0][i].cuda().unsqueeze(0), batch[1][i].cuda().unsqueeze(0)
            pred = model(X)
            predicted.append(pred.argmax(1))
            actual.append(y.item())
            images.append(generate_image(X[0], y.item(), pred.argmax(1).item()))
            pass
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            axs[i, j].imshow(images[i*8+j], cmap=cm.gray)
            axs[i, j].set_title(classes[predicted[i*8+j].item()])
            axs[i, j].axis('off')
            axs[i, j].title.set_size(8)
    plt.ion()
    plt.pause(0.001)


def main(params):
    # Download training data from open datasets.
    training_data = datasets.StanfordCars(
        root="data",
        split='train',
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.StanfordCars(
        root="data",
        split='test',
        download=True,
        transform=ToTensor(),
    )

    model = NeuralNetwork(params['input_size'], params['layers'], params['output_size']).cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])

    train_dataloader = DataLoader(training_data, batch_size=params['batch_size'])
    test_dataloader = DataLoader(test_data, batch_size=params['batch_size'])

    loss_results = []
    accuracy_results = []

    print("[OK] Training model...")
    for t in range(params['epochs']):
        print(f"[OK] Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimiser)
        a, l = test(test_dataloader, model, loss_fn)
        accuracy_results.append(a)
        loss_results.append(l)

    print("[OK] Plotting results...")
    plot_results(accuracy_results, loss_results)
    print("[OK] Done!")

    print("[OK] Done!")

    print("[OK] Saving model...")
    if not os.path.exists(params["model_name"] + ".pth"):
        torch.save(model.state_dict(), f'{params["model_name"]}.pth')
        print("[OK] Model saved as", f'{params["model_name"]}.pth')

        print("[OK] Reloading model...")
        model = NeuralNetwork(params['input_size'], params['layers'], params['output_size']).cuda()
        model.load_state_dict(torch.load(f'{params["model_name"]}.pth'))
        print("[OK] Model reloaded!")
    else:
        print("[ERROR] Model already exists!")

    print("[OK] Testing model...")
    test(test_dataloader, model, loss_fn)
    print("[OK] Done!")

    print("[OK] Checking 9 random images...")
    check_64(test_dataloader, model, params)
    print("[OK] Done!")

    input("Press Enter to end...")
    pass


def parse_parameters(argv):
    if len(argv) < 3 or len(argv) > 8:
        print("Usage: python3 NN_Testing.py <model_name> <layers> <batch_size=64> <epochs=10> <learning_rate=0.0001> <input_size=28*28> <output_size=10>")
        sys.exit(1)
    model_name = argv[1]
    print("[OK] model name =", model_name)
    layers = argv[2].split(',')
    if layers[0][0] == '[':
        layers[0] = layers[0][1:]
    if layers[-1][-1] == ']':
        layers[-1] = layers[-1][:-1]
    batch_size = 64
    epochs = 10
    learning_rate = 0.0001
    input_size = 28 * 28
    output_size = 10
    layers = [int(layer) for layer in layers]
    print("[OK] layers =", layers)
    if len(argv) >= 4:
        batch_size = int(argv[3])
        print("[OK] batch_size =", batch_size)
    if len(argv) >= 5:
        epochs = int(argv[4])
        print("[OK] epochs =", epochs)
    if len(argv) >= 6:
        learning_rate = float(argv[7])
        print("[OK] learning_rate =", learning_rate)
    if len(argv) >= 7:
        input_size = int(argv[6])
        print("[OK] input_size =", input_size)
    if len(argv) >= 8:
        output_size = int(argv[7])
        print("[OK] output_size =", output_size)
    params = {
        'model_name': model_name,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'input_size': input_size,
        'output_size': output_size,
        'layers': layers
    }
    print("[OK] Creating model with parameters:")
    for key, value in params.items():
        print(f"    {key} = {value}")
    main(params)


if __name__ == '__main__':
    parse_parameters(sys.argv)