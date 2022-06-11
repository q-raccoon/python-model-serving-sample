import numpy as np
import tensorflow as tf
from .efficientdet_lite_v4_handler import EfficientDetLiteV4Handler
from ..inference import Inference
from PIL import Image

class EfficientDetLiteV4Inference(Inference):
    def __init__(self) -> None:
        self.handler_ = EfficientDetLiteV4Handler()
        super().__init__()

    def call(self, image: Image.Image, draw_image: bool=True):
        preprocessed_image = self.preprocess(image)
        predictions =  self.handler_(preprocessed_image)
        bboxes, _, _ = self.postprocess(predictions)

        if draw_image:
            self.draw_bounding_boxes(images=preprocessed_image, bboxes=bboxes)
        return bboxes

    def preprocess(self, image: Image.Image) -> tf.image:
        resized_image = np.expand_dims(np.array(image.resize((640, 640))), axis=0)
        tf_image = tf.image.convert_image_dtype(resized_image, dtype=tf.uint8)
        return tf_image

    def postprocess(self, predictions: dict) -> tf.Tensor:
        boxes = predictions["output_0"][0]
        scores = predictions["output_1"][0]
        classes = predictions["output_2"][0]
        
        selected_boxes_index = tf.image.non_max_suppression(boxes, scores, 3)
        selected_boxes = tf.gather(boxes, selected_boxes_index)
        selected_scores = tf.gather(scores, selected_boxes_index)
        selected_classes = tf.gather(classes, selected_boxes_index)

        selected_boxes = tf.expand_dims(selected_boxes, axis=0)
        selected_scores = tf.expand_dims(selected_scores, axis=0)
        selected_classes = tf.expand_dims(selected_classes, axis=0)
        return selected_boxes, selected_scores, selected_classes
    
    def draw_bounding_boxes(self, images: tf.image, bboxes: tf.Tensor):
        float_images = tf.cast(images, dtype=tf.float32)
        bboxes /= 640.0
        colors = np.array([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0]])
        visualized_image = tf.image.draw_bounding_boxes(float_images, bboxes, colors)
        visualized_np_image = tf.cast(tf.squeeze(visualized_image, axis=0), dtype=tf.uint8).numpy()
        
        pil_image = Image.fromarray(visualized_np_image)
        pil_image.save("./test_result.png")

    def __call__(self, image: np.array):
        return self.call(image)