from glob import glob
from process.preprocess import normalize
from loguru import logger
from model.mnist.mnist_handler import MnistHandler
import numpy as np

mnist_handler = None
def get_predictions(image):
    global mnist_handler

    if mnist_handler is None:
        mnist_handler = MnistHandler()

    logger.debug("Image: [{}]".format(image.shape))
    normalized_image = normalize(image)

    logits = mnist_handler.handler(normalized_image)["dense_1"]
    max_logits = int(np.argmax(logits[0]))
    confidence = float(logits[0][max_logits])
    logger.info("predict: {}, confidence: {}".format(max_logits, confidence))
    return max_logits, confidence