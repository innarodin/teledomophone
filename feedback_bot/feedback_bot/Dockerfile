FROM ubuntu:16.04 

LABEL maintainer="rodinskayaia"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        python3 \
        python3-dev \ 
        rsync \
        software-properties-common \
        unzip \
	postgresql \
	postgresql-contrib \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV TZ=Asia/Yekaterinburg

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install \
	pika \
	configobj \
	py-postgresql \
	pytelegrambotapi

RUN apt-get update && apt-get install -y git
RUN pip3 --no-cache-dir install pytz

# TensorBoard
EXPOSE 6012

WORKDIR "/app"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
