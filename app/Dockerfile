FROM python:3.6.9

LABEL maintainer="https://github.com/lyzqm123"
LABEL version="0.1"
LABEL description="Model serving using `Tensorflow` and `FastAPI`"

ENV MODEL_EXPORT_PATH="/code/mnist/saved_model"

WORKDIR /code

COPY ./app/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

WORKDIR /code/app

CMD /bin/bash /code/app/run.sh
