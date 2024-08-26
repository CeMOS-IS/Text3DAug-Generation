# Dockerfile, Image, Container
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# APT packages
RUN apt-get update && apt-get upgrade -y
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt install libxext6 -y
RUN apt install libsm6 -y
RUN apt install ffmpeg -y
RUN apt install nano -y
RUN apt install git -y
RUN apt install lsof -y
RUN apt install pip -y
RUN apt install git-lfs -y
RUN apt install libsparsehash-dev -y
RUN apt install build-essential -y
RUN apt install wget -y
RUN apt-get install libosmesa6-dev -y


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl \
    libnvidia-gl-535-server
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics


# Python Packages
RUN python3 -m pip install --upgrade pip
RUN pip install --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Setup directories
COPY ./ /home/
WORKDIR /home/

# Setup local packages
RUN pip install -e packages/point-e --user
RUN pip install -e packages/shap-e --user
RUN pip install -e packages/cap3d --user
RUN pip install -r ./setup/requirements.txt

# Setup taming
RUN pip install -e git+https://github.com/CompVis/taming-transformers.git\#egg=taming-transformers --user

# Entrypoint downloads weights and compiles GPU support packages (e.g. PyTorch3D)
# This can take a long time when running the container for the first time (> 30 mins)
RUN chmod +x ./setup/docker_setup.sh
ENTRYPOINT ["./setup/docker_setup.sh"]
