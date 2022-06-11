from fastapi import APIRouter
from fastapi import UploadFile, File
from predict import get_predictions
from data.dataset import Dataset
from .read_file import read_imagefile

router = APIRouter()

@router.post("/predict/image_classification", tags=["image_classification"])
async def predict_api(dataset: Dataset = Dataset.mnist, file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    pil_image = read_imagefile(dataset, await file.read())

    logits, confidence = get_predictions(dataset, pil_image)
    return {"predict": logits, "confidence": confidence}
