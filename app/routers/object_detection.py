from fastapi import APIRouter
from fastapi import UploadFile, File
from predict import get_predictions
from data.dataset import Dataset
from .read_file import read_imagefile

router = APIRouter()

@router.post("/predict/object_detection", tags=["object_detection"])
async def predict_api(dataset: Dataset = Dataset.coco2017, file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    pil_image = read_imagefile(dataset, await file.read())

    bboxes_and_labels = get_predictions(dataset, pil_image)
    return {"bboxes_and_labels": bboxes_and_labels}
