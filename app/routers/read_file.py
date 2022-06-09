from fastapi import UploadFile
from PIL import Image, ImageOps
from io import BytesIO
from data.dataset import Dataset
import numpy as np

def read_imagefile(dataset: Dataset, file: UploadFile) -> Image.Image:
    image = Image.open(BytesIO(file))
    if dataset == Dataset.mnist:
        image = ImageOps.grayscale(image)
    np_image = np.asarray(image)
    return np_image