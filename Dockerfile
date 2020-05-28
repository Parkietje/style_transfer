FROM python:3.8-slim

RUN apt-get update

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY style-transfer.py .

ENTRYPOINT ["python", "style-transfer.py"]