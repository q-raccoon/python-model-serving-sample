from abc import ABCMeta, abstractmethod
from PIL import Image


class Inference(metaclass=ABCMeta):
    def __init__(self):
        super(Inference, self).__init__()

    @abstractmethod
    def call(self, image: Image.Image):
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image):
        pass
