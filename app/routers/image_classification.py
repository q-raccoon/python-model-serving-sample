from fastapi import APIRouter
from fastapi import UploadFile, File
from PIL import Image, ImageOps
from io import BytesIO
from predict import get_predictions
from data.dataset import Dataset
import numpy as np

router = APIRouter()


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    gray_image = ImageOps.grayscale(image)
    np_image = np.asarray(gray_image)
    return np_image


@router.post("/predict/image", tags=["image_classification"])
async def predict_api(dataset: Dataset = Dataset.mnist, file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    np_image = read_imagefile(await file.read())

    logits, confidence = get_predictions(dataset, np_image)
    return {"predict": logits, "confidence": confidence}
