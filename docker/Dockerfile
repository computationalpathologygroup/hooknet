FROM nvidia/cuda:11.1-devel-ubuntu20.04

# Set time zone
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-dev -y \
    && apt-get clean \
    && :

# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install ASAP2.0
RUN : \
    && apt-get update \
    && apt-get -y install curl git \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.0-(Nightly)/ASAP-2.0-py38-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.0-py38-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
    && rm ASAP-2.0-py38-Ubuntu2004.deb \
    && :

# Install zip
RUN apt-get install zip --assume-yes

# Install HookNet from Github
ARG hooknet=https://github.com/computationalpathologygroup/hooknet/archive/master.zip
ADD ${hooknet} hooknet.zip
RUN unzip hooknet.zip -d /venv/lib/python3.8/site-packages/
RUN mv /venv/lib/python3.8/site-packages/hooknet-master /venv/lib/python3.8/site-packages/hooknet
RUN rm hooknet.zip


# Install requirements
COPY ./requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/

# Set workdir
WORKDIR /venv/lib/python3.8/site-packages/hooknet