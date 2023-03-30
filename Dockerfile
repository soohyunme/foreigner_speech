FROM flml/flashlight:cuda-latest

# Install basic packages
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="Asia/Seoul" apt-get -y install \
        tmux \
        gdb \
        curl \
        rsync \
        rename \
        ffmpeg \
        htop && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    ln -s /usr/bin/python3.8 /usr/bin/python

# upgrade pip 
RUN pip install --upgrade pip

# install etc modules 
RUN pip install soundfile \
    packaging \
    torch \
    editdistance \
    tensorboardX \
    nlptutti

# Install language setup in Korean 
RUN apt-get update && apt-get install locales && \
    locale-gen ko_KR && \
    echo "export LC_ALL=ko_KR.UTF-8" >> ~/.bashrc

# Install flashlight 
WORKDIR /root/flashlight/bindings/python
RUN pip install -e .

# Install ctcdecoder 
WORKDIR /opt
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
WORKDIR /opt/ctcdecode
RUN pip install .

# Hydra setting 
ENV HYDRA_FULL_ERROR=1

WORKDIR /root