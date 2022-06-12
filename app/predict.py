from loguru import logger
from model.mnist.mnist_inference import MnistInference
from model.coco2017.efficientdet_lite_v4_inference import EfficientDetLiteV4Inference
from data.dataset import Dataset
from PIL import Image

mnist_inference = None
efficient_det_lite_v4_inference = None


def get_predictions(dataset: Dataset = Dataset.mnist, image: Image.Image = None):

    inference = None
    if dataset == Dataset.mnist:
        global mnist_inference
        if mnist_inference is None:
            mnist_inference = MnistInference()
        inference = mnist_inference
    elif dataset == Dataset.coco2017:
        global efficient_det_lite_v4_inference
        if efficient_det_lite_v4_inference is None:
            efficient_det_lite_v4_inference = EfficientDetLiteV4Inference(draw_image=False)
        inference = efficient_det_lite_v4_inference
    else:
        raise RuntimeError("`{}` is not supported yet.".format(dataset))

    try:
        logger.debug("Ipunt image height: {}, width: {}".format(
            image.height, image.width))
        predictions = inference(image)
    except Exception as e:
        raise RuntimeError(e)

    logger.info("{}".format(predictions))
    return predictions
