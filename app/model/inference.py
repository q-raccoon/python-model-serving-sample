from abc import ABCMeta, abstractmethod
from typing import Union
import numpy as np


class Inference(metaclass=ABCMeta):
    def __init__(self):
        super(Inference, self).__init__()

    @abstractmethod
    def call(self, image: np.array):
        pass

    @abstractmethod
    def preprocess(self, image: np.array) -> np.array:
        pass
