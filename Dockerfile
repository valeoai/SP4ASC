FROM nvidia/cuda:10.2-devel

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#dockerfiles
# With these variables you don't have to run the final image with the --gpus argument.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && apt-get install -y sudo

WORKDIR /root

RUN sudo apt-get update && sudo apt-get install -y openssh-server
COPY id_rsa.pub /root/.ssh/authorized_keys

RUN chmod 400 ~/.ssh/authorized_keys

RUN apt update \
    && apt install -y python3.7 python3-pip git \
    && python3.7 -m pip install --upgrade --force pip
RUN ln -s /usr/bin/python3.7 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

RUN sudo apt install -y tmux
RUN sudo apt install -y nano

# --- Python environment
RUN pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install tqdm scikit-learn tensorboard pandas pyaml torchlibrosa
RUN apt install -y libsndfile1
