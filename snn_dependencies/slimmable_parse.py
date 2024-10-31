import sys
import json

from orca.orca import start


def print_help():
    print("Usage: python3 snn.py <model_name> [-d <dataset>] [-w <width_mult_list>] [-l <layer_number>] [-k <kernel_size>] [-s <stride>] [-p] [-dv <device>] [-lr <learning_rate>] [-b <batch_size>] [-e <epochs>] [-v | -vv | -vvv]")
    print("Arguments:")
    print("    - model_name: name of the model")
    print("    - dataset: one of the following datasets: CIFAR10, CIFAR100, MNIST (default: CIFAR10)")
    print("    - width_mult_list: list of width multipliers (default: [0.25, 0.5, 0.75, 1.0])")
    print("    - layer_number: number of layers in the model (default: 6)")
    print("    - kernel_size (default: 3)")
    print("    - stride (default: 1)")
    print("    - argument for pretrained model w/ model file (default: False) (flag: -p)")
    print("    - learning_rate (default: 0.001)")
    print("    - batch_size (default: 64)")
    print("    - epochs (default: 10)")
    print("    - verbosity (default: 0) (flag: -v | -vv | -vvv)")
    print("    - device (default: 'cuda') (flag: -dv)")
    print("    - help: -h (prints help message)")

def parse_params(argv):
    if len(argv) < 2:
        print("[ERROR] Missing model name")
        print_help()
        sys.exit(1)

    if "-h" in argv:
        print_help()
        sys.exit(0)

    model_name = argv[1]

    dataset = "CIFAR10"
    input_size = 32 * 32
    output_size = 10
    in_channels = 9
    try:
        if "-d" in argv:
            dataset = argv[argv.index("-d") + 1]
            if dataset.upper() not in ["CIFAR10", "CIFAR100", "MNIST"]:
                print("[WARNING] Invalid dataset given, defaulting to CIFAR10")
                dataset = "CIFAR10"
    except IndexError:
        print("[WARNING] Dataset not given when flag raised, defaulting to CIFAR10")
        dataset = "CIFAR10"
    match dataset.upper():
        case "CIFAR10":
            input_size = 32 * 32
            output_size = 10
            in_channels = 9
        case "CIFAR100":
            input_size = 32 * 32
            output_size = 100
            in_channels = 9
        case "MNIST":
            input_size = 28 * 28
            output_size = 10
            in_channels = 1
        case _:
            print("[ERROR] Invalid dataset somehow parsed: ", dataset)

    width_mult_list = [0.25, 0.5, 0.75, 1.0]
    try:
        if "-w" in argv:
            width_mult_list = list(map(float, argv[argv.index("-w") + 1].split(",")))
            if not all([0 < i <= 1 for i in width_mult_list]):
                print("[WARNING] Invalid width multiplier list given, defaulting to [0.25, 0.5, 1.0]")
                width_mult_list = [0.25, 0.5, 0.75, 1.0]
    except IndexError:
        print("[WARNING] Width multiplier list not given when flag raised, defaulting to [0.25, 0.5, 1.0]")
        width_mult_list = [0.25, 0.5, 0.75, 1.0]
    except ValueError:
        print("[WARNING] Invalid width multiplier list given, defaulting to [0.25, 0.5, 1.0]")
        width_mult_list = [0.25, 0.5, 0.75, 1.0]

    pretrained = False
    if "-p" in argv:
        pretrained = True
        print("[OK] Pretrained model being used...")

    learning_rate = 0.001
    try:
        if "-lr" in argv:
            learning_rate = float(argv[argv.index("-lr") + 1])
            if learning_rate <= 0:
                print("[WARNING] Invalid learning rate given, defaulting to 0.001")
                learning_rate = 0.001
    except IndexError:
        print("[WARNING] Learning rate not given when flag raised, defaulting to 0.001")
        learning_rate = 0.001
    except ValueError:
        print("[WARNING] Invalid learning rate given, defaulting to 0.001")
        learning_rate = 0.001

    batch_size = 64
    try:
        if "-b" in argv:
            batch_size = int(argv[argv.index("-b") + 1])
            if batch_size <= 0:
                print("[WARNING] Invalid batch size given, defaulting to 64")
                batch_size = 64
    except IndexError:
        print("[WARNING] Batch size not given when flag raised, defaulting to 64")
        batch_size = 64
    except ValueError:
        print("[WARNING] Invalid batch size given, defaulting to 64")
        batch_size = 64

    layer_num = 6
    try:
        if "-l" in argv:
            layer_num = int(argv[argv.index("-l") + 1])
            if layer_num <= 0:
                print("[WARNING] Invalid layer number given, defaulting to 4")
                layer_num = 6
    except IndexError:
        print("[WARNING] Layer number not given when flag raised, defaulting to 4")
        layer_num = 6
    except ValueError:
        print("[WARNING] Invalid layer number given, defaulting to 4")
        layer_num = 6

    max_channels_per_layer_list = []
    channel_max = 512
    max_channels_per_layer_list = [128, 256, 512]
    start_size = len(max_channels_per_layer_list)
    for i in range(start_size,  layer_num):
        max_channels_per_layer_list.append(channel_max)

    epochs = 10
    try:
        if "-e" in argv:
            epochs = int(argv[argv.index("-e") + 1])
            if epochs <= 0:
                print("[WARNING] Invalid epoch count given, defaulting to 10")
                epochs = 10
    except IndexError:
        print("[WARNING] Epoch count not given when flag raised, defaulting to 10")
        epochs = 10
    except ValueError:
        print("[WARNING] Invalid epoch count given, defaulting to 10")
        epochs = 10

    verbosity = 0
    if "-v" in argv:
        verbosity = 1
    elif "-vv" in argv:
        verbosity = 2
    elif "-vvv" in argv:
        verbosity = 3

    stride = 1
    try:
        if "-s" in argv:
            stride = int(argv[argv.index("-s") + 1])
            if stride <= 0:
                print("[WARNING] Invalid stride given, defaulting to 1")
                stride = 1
    except IndexError:
        print("[WARNING] Stride not given when flag raised, defaulting to 1")
        stride = 1
    except ValueError:
        print("[WARNING] Invalid stride given, defaulting to 1")
        stride = 1

    kernel_size = 3
    try:
        if "-k" in argv:
            kernel_size = int(argv[argv.index("-k") + 1])
            if kernel_size <= 0:
                print("[WARNING] Invalid kernel size given, defaulting to 3")
                kernel_size = 3
    except IndexError:
        print("[WARNING] Kernel size not given when flag raised, defaulting to 3")
        kernel_size = 3
    except ValueError:
        print("[WARNING] Invalid kernel size given, defaulting to 3")
        kernel_size = 3

    device = "cuda"
    try:
        if "-dv" in argv:
            device = int(argv[argv.index("-dv") + 1])
            if device != "cuda" or device != "cpu":
                print("[WARNING] Invalid device given, defaulting to cuda")
                device = "cuda"
    except IndexError:
        print("[WARNING] Device not given when flag raised, defaulting to cuda")
        device = "cuda"

    params = {
        'model_name': model_name,
        'dataset': dataset,
        'input_size': input_size,
        'output_size': output_size,
        'width_mult_list': width_mult_list,
        'layer_num': layer_num,
        'max_channels_per_layer_list': max_channels_per_layer_list,
        'kernel_size': kernel_size,
        'stride': stride,
        'pretrained': pretrained,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'verbosity': verbosity,
        'in_channels': in_channels,
        'device': device
    }

    if verbosity > 1: print("[OK] Saving parameters to file: snn_models/" + model_name + ".json...")
    # Save parameters to file (for easy access):
    with open("snn_models/" + model_name + ".json", "w") as f:
        json.dump(params, f)
    if verbosity > 1: print("[OK] Done!")

    if verbosity > 0:
        print("[OK] Creating model with parameters:")
        for key, value in params.items():
            print(f"    {key} = {value}")

    return params