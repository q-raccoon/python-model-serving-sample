import numpy as np
import tensorflow as tf
from .efficientdet_lite_v4_handler import EfficientDetLiteV4Handler
from ..inference import Inference


class EfficientDetLiteV4Inference(Inference):
    def __init__(self) -> None:
        self.handler_ = EfficientDetLiteV4Handler()
        super().__init__()

    def call(self, image: np.array):
        preprocessed_image = self.preprocess(image)
        results =  self.handler_(preprocessed_image)
        return results

    def preprocess(self, image: np.array) -> tf.image:
        resized_image = np.resize(image, (1, 640, 640, 3))
        tf_image = tf.image.convert_image_dtype(resized_image, dtype=tf.uint8)
        return tf_image

    def __call__(self, image: np.array):
        return self.call(image)