from predict import get_predictions
from loguru import logger
from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np

app = FastAPI()

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    gray_image = ImageOps.grayscale(image)
    np_image = np.asarray(gray_image)
    return np_image

@app.get("/")
async def read_root():
    return {"Predict"}

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    np_image = read_imagefile(await file.read())

    logits, confidence = get_predictions(np_image)
    return {"predict": logits, "confidence": confidence}
