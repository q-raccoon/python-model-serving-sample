from process.preprocess import normalize
from loguru import logger
from model.mnist.mnist_handler import MnistHandler
from data.dataset import Dataset
import numpy as np

mnist_handler = None


def get_predictions(dataset: Dataset = Dataset.mnist, image: np.array = None):

    handler = None
    if dataset == Dataset.mnist:
        global mnist_handler
        if mnist_handler is None:
            mnist_handler = MnistHandler()
        handler = mnist_handler
    else:
        raise RuntimeError("`{}` is not supported yet.".format(dataset))
        
    try:
        logger.debug("Image: [{}]".format(image.shape))
        normalized_image = normalize(image)
        predictions = handler.handler(normalized_image)
    except Exception as e:
        raise RuntimeError(e)


    logger.info("{}".format(predictions))
    return predictions
