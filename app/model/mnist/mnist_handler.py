import tensorflow as tf
import numpy as np
from model.handler import Handler
from loguru import logger
import os

class MnistHandler(Handler):
    def __init__(self) -> None:
        self.export_path = os.environ.get("MODEL_EXPORT_PATH", "/tmp/mnist/saved_model")
        if not os.path.exists(self.export_path):
            raise RuntimeError("Model export path is not exist. [{}]".format(self.export_path))
        self.serving_fn = (tf.keras.models.load_model(self.export_path).signatures["serving_default"])
        self.output_keys = [key for key in self.serving_fn.structured_outputs.keys()]
        logger.info("`serving_fn` is updated. export path: {}, output keys: {}".format(self.export_path, self.output_keys))
        super().__init__()

    def handler(self, request):
        # [TODO] Fix the specific argument of `serving_fn` method.
        predictions = self.serving_fn(input_1=request)[self.output_keys[0]]
        
        prediction = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][prediction])
        return prediction, confidence