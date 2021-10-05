# Pull Base Image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set Working Directory
RUN mkdir /usr/src/point-cloud-segmentation
WORKDIR /usr/src/point-cloud-segmentation

# Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.docker /usr/src/point-cloud-segmentation/requirements.docker

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.docker

COPY . /usr/src/point-cloud-segmentation/

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
