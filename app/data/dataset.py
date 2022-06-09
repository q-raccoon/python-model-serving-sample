from enum import Enum

class Dataset(str, Enum):
    mnist = "MNIST"
    coco2017 = "COCO2017"