import tensorflow as tf
import numpy as np
from model.handler import Handler
from loguru import logger
import os


class EfficientDetLiteV4Handler(Handler):
    def __init__(self) -> None:
        self.export_path = os.environ.get(
            "DET_MODEL_EXPORT_PATH", "/tmp/coco2017/efficientdet_lite4_detection")
        if not os.path.exists(self.export_path):
            raise RuntimeError(
                "Model export path is not exist. [{}]".format(self.export_path))
        self.serving_fn = (tf.keras.models.load_model(
            self.export_path).signatures["serving_default"])
        self.output_keys = [
            key for key in self.serving_fn.structured_outputs.keys()]
        logger.info("`serving_fn` is updated. export path: {}, output keys: {}".format(
            self.export_path, self.output_keys))
        super().__init__()

    def handle(self, request):
        """ 
            predictions: {
                            'output_0': boxes ([batch, num_detections, 4]), 
                            'output_1': scores ([batch, num_detections]), 
                            'output_2': classes ([batch, num_detections]), 
                            'output_3': num_detections (1)
                         }
        """
        predictions = self.serving_fn(request)
        return predictions

    def __call__(self, request):
        return self.handle(request)