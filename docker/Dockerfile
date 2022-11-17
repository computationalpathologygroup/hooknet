FROM tensorflow/tensorflow:2.9.0-gpu


# Install ASAP 2.1
RUN : \
    && apt-get update \
    && apt-get -y install curl \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.1-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /usr/local/lib/python3.8/dist-packages/asap.pth \
    && rm ASAP-2.1-Ubuntu2004.deb \
    && :

# Install zip
RUN apt-get install zip --assume-yes

# Install HookNet from Github
ARG hooknet=https://github.com/computationalpathologygroup/hooknet/archive/master.zip
ADD ${hooknet} hooknet.zip
RUN unzip hooknet.zip -d /usr/local/lib/python3.8/dist-packages/
RUN mv /usr/local/lib/python3.8/dist-packages/hooknet-master /usr/local/lib/python3.8/dist-packages/hooknet
RUN rm hooknet.zip

# Install requirements
COPY ./requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/

# Set workdir
WORKDIR /usr/local/lib/python3.8/dist-packages/hooknet
