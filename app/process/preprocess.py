from ctypes import resize
import numpy as np

def normalize(image):
    resized_image = np.resize(image, (28,28,1))

    flatten_image = resized_image.flatten()

    flatten_image = 255 - flatten_image
    normalized_image = flatten_image / float(255)

    expanded_image = np.expand_dims(normalized_image, 0)
    return expanded_image.astype(dtype=np.float32)