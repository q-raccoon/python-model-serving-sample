from loguru import logger
from model.mnist.mnist_inference import MnistInference
from data.dataset import Dataset
import numpy as np

mnist_inference = None


def get_predictions(dataset: Dataset = Dataset.mnist, image: np.array = None):

    inference = None
    if dataset == Dataset.mnist:
        global mnist_inference
        if mnist_inference is None:
            mnist_inference = MnistInference()
        inference = mnist_inference
    else:
        raise RuntimeError("`{}` is not supported yet.".format(dataset))
        
    try:
        logger.debug("Image: [{}]".format(image.shape))
        predictions = inference(image)
    except Exception as e:
        raise RuntimeError(e)


    logger.info("{}".format(predictions))
    return predictions
