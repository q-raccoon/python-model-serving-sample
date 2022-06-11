import numpy as np
from PIL import Image
from .mnist_handler import MnistHandler
from ..inference import Inference


class MnistInference(Inference):
    def __init__(self) -> None:
        self.handler_ = MnistHandler()
        super().__init__()

    def call(self, image: np.array):
        preprocessed_image = self.preprocess(image)
        results =  self.handler_(preprocessed_image)
        return results

    def preprocess(self, image: Image.Image) -> np.array:
        resized_image = np.array(image.resize((28, 28)))
        flatten_image = resized_image.flatten()
        flatten_image = 255.0 - flatten_image
        normalized_image = flatten_image / 255.0
        expanded_image = np.expand_dims(normalized_image, 0)
        return expanded_image.astype(dtype=np.float32)

    def __call__(self, image: np.array):
        return self.call(image)