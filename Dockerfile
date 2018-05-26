FROM ubuntu:latest

WORKDIR ./home

COPY . ./
RUN apt-get update && \
    apt-get -y install python-pip 

RUN pip install -r requirements.txt
