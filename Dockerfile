FROM docker.io/continuumio/miniconda3:latest

# RUN apt update && apt install python3-opencv
# RUN apt update && apt install libgl1
RUN apt update && apt install ffmpeg libsm6 libxext6 -y

WORKDIR /app
COPY . .

RUN conda env create -n drowsy --file drowsy.yml