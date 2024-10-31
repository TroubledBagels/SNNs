# Slimmable Implementation

This is an implementation of the basic slimmable neural network (SNN) originally suggested by Yu et al. in their paper [Slimmable Neural Networks](https://arxiv.org/abs/1812.08928). The implementation is based on the PyTorch framework and is designed to be as simple and easy to understand as possible. The implementation is also designed to be easily extensible to more complex models and tasks.

## Usage
```python3 snn.py <model_name> [other arguments]```

List of available arguments:
- ```-d <dataset>```: Default is CIFAR10, also available are CIFAR100 and MNIST (not properly implemented yet)
- ```-w <width_mult_list>```: Default is [0.25, 0.5, 0.75, 1.0], give in format "[x, y, z]"
- ```-l <layer_num>```: Default is 6, must be >= 3
- ```-lr <learning_rate>```: Default is 0.001
- ```-b <batch_size>```: Default is 64
- ```-v | -vv | -vvv```: Controls level of verbosity, use none for incredibly low-level
- ```-dv <device>```: Controls the device ("cpu" or "cuda"), default: cuda
- ```-h```: Prints the help message
## Further Explanations

### The data structure for storing weights
#### Basic diagram:

![alt_text](md_assets/Data%20Structure%20for%20Slimmable%20Weights.png)

- Dimension 0: the dimension for concatenating based on the out channels
- Dimension 1: the dimension for concatenating based on the in channels
- Along the bottom is the list of width multipliers used in this example
- Up the left is an example list of possible in_channel numbers

#### Storage:
The storage process for this structure is somewhat strange. It uses a python list that stores ```w0``` followed by a series of pairs ```(wn.1, wn.2)```, i.e. like the following: ```[w0, (w1.1, w1.2), ..., (wn.1, wn.2)]```.
