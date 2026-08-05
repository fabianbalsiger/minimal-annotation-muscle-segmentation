# FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04
FROM nvidia/cuda:13.3.1-runtime-ubuntu26.04

LABEL application="Muscle segmentation training using the nnU-Net V2"
LABEL author="Fabian Balsiger and Pierre-Yves Baudin"

ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /

# SYSTEM
RUN apt update --yes --quiet && apt install --yes --quiet --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends \
    python3.10 libpython3.10-dev python3.10-distutils build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Switch default Python version to 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2 && \
 update-alternatives --set python /usr/bin/python3.10 && \
 update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 && \
 update-alternatives --set python3 /usr/bin/python3.10

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# Copy necessary files to the working directory
COPY ./requirements.txt .
COPY ./train.sh .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy custom trainer and set nnU-Net environment variable
ENV nnUNet_raw="/nnunet/nnUNet_raw"
ENV nnUNet_results="/nnunet/nnUNet_results"
ENV nnUNet_preprocessed="/nnunet/nnUNet_preprocessed"

ENTRYPOINT ["bash"]
CMD ["train.sh"]
