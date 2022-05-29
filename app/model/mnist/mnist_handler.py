import tensorflow as tf
from model.handler import Handler
from loguru import logger
import os

class MnistHandler(Handler):
    def __init__(self) -> None:
        self.export_path = os.path.join(os.getcwd(), "model/mnist/saved_model")
        self.serving_fn = (tf.keras.models.load_model(self.export_path).signatures["serving_default"])
        logger.info("`serving_fn` is updated. export path: {}".format(self.export_path))
        super().__init__()

    def handler(self, request):
        return self.serving_fn(reshape_input=request)