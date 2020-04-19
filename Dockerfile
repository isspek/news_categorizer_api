FROM tiangolo/uvicorn-gunicorn:python3.7

COPY . /app
WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

