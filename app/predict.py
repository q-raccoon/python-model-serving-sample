from process.preprocess import normalize
from loguru import logger
from model.mnist.mnist_handler import MnistHandler

mnist_handler = None
def get_predictions(image):
    global mnist_handler

    if mnist_handler is None:
        mnist_handler = MnistHandler()

    logger.debug("Image: [{}]".format(image.shape))
    normalized_image = normalize(image)

    predictions = mnist_handler.handler(normalized_image)

    logger.info("{}".format(predictions))
    return predictions