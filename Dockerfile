FROM python:3

WORKDIR ./home

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT [ "bash" ]
