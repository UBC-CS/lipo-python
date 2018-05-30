FROM python:3

COPY . ./

RUN pip install -r requirements.txt 

WORKDIR ./home

ENTRYPOINT [ "bash" ]
