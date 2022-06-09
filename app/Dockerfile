FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="https://github.com/lyzqm123"
LABEL version="0.1"
LABEL description="Model serving using `Tensorflow` and `FastAPI`"

ENV MODEL_EXPORT_PATH="/code/mnist/saved_model"
ENV DET_MODEL_EXPORT_PATH="/code/coco2017/efficientdet_lite4_detection_2"

WORKDIR /code

COPY ./mnist /code/mnist
COPY ./code/coco2017/ /code/coco2017

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./ /code/app

WORKDIR /code/app

CMD /bin/bash /code/app/run.sh
