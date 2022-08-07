import time
import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'routers'))
from routers import image_classification, object_detection
from loguru import logger
from fastapi import FastAPI, Request
import numpy as np
app = FastAPI()
app.include_router(image_classification.router)
app.include_router(object_detection.router)

@app.middleware("http")
async def check_inference_latency(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    inference_latency = time.perf_counter() - start_time
    logger.info("inference latency: %.6f (sec)" % (inference_latency))
    return response

@app.get("/")
def read_root():
    return {"msg": "This is MLops project."}