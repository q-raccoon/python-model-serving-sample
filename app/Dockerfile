FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="https://github.com/lyzqm123"
LABEL version="0.1"
LABEL description="Model serving using `Tensorflow` and `FastAPI`"

ENV MODEL_EXPORT_PATH="/data1/mnist/saved_model"
ENV DET_MODEL_EXPORT_PATH="/data2/coco2017/efficientdet_lite4_detection"

WORKDIR /code

COPY ./app/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

WORKDIR /code/app

CMD /bin/bash /code/app/run.sh