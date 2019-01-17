FROM mxnet_opencv_image

LABEL maintainer="rodinskayaia"

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install \
	pika \
	redis \
	python-logstash

RUN apt-get update && apt-get install -y git

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ENV TZ=Asia/Yekaterinburg

# TensorBoard
EXPOSE 6006

WORKDIR "/app"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
