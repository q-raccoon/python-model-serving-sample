import tensorflow as tf
from abc import ABCMeta, abstractmethod

class ExportModel(metaclass=ABCMeta):
    def __init__(self):
        super(ExportModel, self).__init__()

    @abstractmethod
    def serving_fn(self, inputs):
        pass

if __name__ == "__main__":
    pass
