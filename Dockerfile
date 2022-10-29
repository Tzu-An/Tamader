FROM python:3.8-bullseye

RUN apt-get update && apt-get upgrade
RUN mkdir -p /home/app
ENV ENV_VAR=prod

WORKDIR /home/app
RUN mkdir scripts configs

COPY app.py .
COPY scripts scripts
COPY configs configs
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]

EXPOSE 5000
