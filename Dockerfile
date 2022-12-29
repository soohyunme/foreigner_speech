FROM flml/flashlight:cuda-latest

# Install basic packages
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="Asia/Seoul" apt-get -y install \
        cmake \
        gcc \
        g++ \
        wget \
        vim \
        tmux \
        gdb \
        curl \
        rsync \
        rename \
        sox \
        ffmpeg \
        openssh-server \
        git \
        nano \
        htop \
        python3.8 \
        python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# upgrade pip 
RUN pip install --upgrade pip

# install etc modules 
RUN pip install soundfile \
    packaging \
    torch \
    editdistance \
    tensorboardX

# Install kenlm 
WORKDIR /opt
RUN apt-get update && wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz && \
        mkdir kenlm/build && cd kenlm/build && \
        cmake .. -DCMAKE_BUILD_TYPE=Release \
                 -DCMAKE_INSTALL_PREFIX=/opt/kenlm \
                 -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
        make -j $(nproc)

# Install flashlight 
WORKDIR /root
RUN apt-get update && cd flashlight/bindings/python && \
        pip install -e .

# Install language setup in Korean 
RUN apt-get update && apt-get install locales && \
    locale-gen ko_KR && \
    echo "export LC_ALL=ko_KR.UTF-8" >> ~/.bashrc

# Install ctcdecoder 
WORKDIR /opt
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode && pip install .

# Hydra setting 
RUN echo "export HYDRA_FULL_ERROR=1" >> ~/.bashrc

# Install SSH related 
RUN echo 'root:qwer12#$' | chpasswd
RUN sed -ri 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
ENTRYPOINT service ssh restart && bash