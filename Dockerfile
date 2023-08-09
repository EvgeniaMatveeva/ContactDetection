# FROM python:3.8.6-buster

FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && \
	apt-get install -y curl python3.8 python3.8-distutils wget && \
	ln -s /usr/bin/python3.8 /usr/bin/python && \
	rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3

ENV PROJECT_ROOT /app

ENV DATA_ROOT /data
ENV OUTPUT_ROOT /output

RUN mkdir $PROJECT_ROOT $DATA_ROOT

COPY . $PROJECT_ROOT

WORKDIR $PROJECT_ROOT

RUN pip install -r requirements.txt

WORKDIR /opt

RUN chmod +x $PROJECT_ROOT$DATA_ROOT/get_model_weights.sh
RUN $PROJECT_ROOT$DATA_ROOT/get_model_weights.sh

WORKDIR $PROJECT_ROOT

ENV TRANSFORMERS_OFFLINE 1

CMD python lib/run.py
