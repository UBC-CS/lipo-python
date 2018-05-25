FROM ubuntu:latest

WORKDIR ./home

COPY . ./
RUN apt-get update && \
    apt-get -y install python-pip && \
    apt-get install python3

RUN pip install -r requirements.txt
