from enum import Enum

class Dataset(str, Enum):
    mnist = "MNIST"
    imagenet = "IMAGENET"