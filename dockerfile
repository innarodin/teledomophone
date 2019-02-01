FROM nvidia/cuda:9.0-base-ubuntu16.04

LABEL maintainer="Yuriy Samusevich"

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        libfreetype6-dev \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3.5 \
        python3.5-dev \
        libsm6 \
        libxrender1 \
        libfontconfig1 \
        libxext6 \
        software-properties-common \
        unzip \
        && \
	apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.5 get-pip.py && \
    rm get-pip.py

RUN pip3.5 --no-cache-dir install \
        configobj \
        easydict \
        h5py \ 
	imgaug \
        imutils \
        matplotlib \
        mxnet-cu90 \
        numpy==1.14.5 \
        opencv-python \
        opencv-contrib-python \
        pandas \
        Pillow \
        pymemcache \
        pytz \
        py-postgresql \
        requests \
        scipy \
        sklearn \
        scikit-image \
        tensorflow_gpu==1.10.1 \
        zmq 
         
RUN apt-get update && apt-get install -y git
      
RUN git clone https://github.com/davidsandberg/facenet

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TZ=Asia/Yekaterinburg
ENV PYTHONPATH='/facenet/src'
