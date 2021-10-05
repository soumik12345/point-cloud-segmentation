# Pull Base Image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set Working Directory
RUN mkdir /usr/src/point-cloud-segmentation
WORKDIR /usr/src/point-cloud-segmentation

# Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.docker /usr/src/point-cloud-segmentation/requirements.docker

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Indian
RUN apt-get update && apt-get install -y \ 
    wget \
    build-essential \ 
    cmake \ 
    git \
    unzip \ 
    pkg-config \
    python-dev \ 
    python-opencv \ 
    libopencv-dev \
    libpng-dev \ 
    libtiff-dev \
    libgtk2.0-dev \ 
    python-numpy \ 
    python-pycurl \ 
    libatlas-base-dev \
    gfortran \
    webp \ 
    python-opencv \ 
    qt5-default \
    libvtk6-dev \ 
    zlib1g-dev 

RUN pip install --upgrade pip setuptools wheel
RUN pip install black==19.10b0 \
    jupyter \
    matplotlib==3.4.3 \
    pandas==1.3.3 \
    plotly==5.3.1 \
    plotly-express==0.4.1 \
    pytest==6.2.5 \
    tqdm==4.62.3

COPY . /usr/src/point-cloud-segmentation/

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
