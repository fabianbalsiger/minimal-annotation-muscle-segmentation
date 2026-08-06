def make_docker(title, outdir, folds=(0, 1, 2, 3, 4), trainer="nnUNetTrainer", dataset="001", config="3d_fullres", nchannel=1):
    """create the dockerfile from the template below"""

    # putting folds in good shape to be processed by CLI :
    foldstr = ", ".join(f'"{f}"' for f in folds)

    # dealing with folder link with the fold number in crossvalidation
    checkpoints_files = ""
    for fold_nbr in folds:
        checkpoints_files = (
            checkpoints_files
            + f"COPY ./nnUNet_results/Dataset001/{trainer}__nnUNetPlans__{config}/fold_{fold_nbr}/checkpoint_final.pth"
            + f" /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__{config}/fold_{fold_nbr}/checkpoint_final.pth\n"
        )

    dockerfile = DOCKER_FILE.format(
        title=title,
        trainer=trainer,
        checkpoints_files=checkpoints_files,
        folds=foldstr,
        dataset=dataset,
        config=config,
        nchannel=nchannel,
    )
    return dockerfile


#
# dockerfile template

DOCKER_FILE = """FROM nvidia/cuda:13.3.1-runtime-ubuntu26.04
LABEL application="Muscle segmentation using nnU-Net V2"
LABEL model=MUSEG
LABEL title={title}
LABEL author="Fabian Balsiger and Pierre-Yves Baudin"
LABEL nchannel={nchannel}


ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /

# SYSTEM
RUN apt update --yes --quiet && apt install --yes --quiet --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends \
    python3.10 libpython3.10-dev python3.10 python3.10-distutils build-essential curl \
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

# Copy the model to the working directory
{checkpoints_files}
COPY ./nnUNet_results/Dataset001/{trainer}__nnUNetPlans__{config}/dataset.json /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__{config}/dataset.json
COPY ./nnUNet_results/Dataset001/{trainer}__nnUNetPlans__{config}/dataset_fingerprint.json /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__{config}/dataset_fingerprint.json
COPY ./nnUNet_results/Dataset001/{trainer}__nnUNetPlans__{config}/plans.json /nnUNet_results/Dataset001/nnUNetTrainer__nnUNetPlans__{config}/plans.json

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Set nnU-Net environment variable
ENV nnUNet_results="/nnUNet_results"
ENV nnUNet_raw="/nnUNet_raw"
ENV nnUNet_preprocessed = "/nnUNet_preprocessed"

COPY ./labels.txt ./labels.txt

ENTRYPOINT ["nnUNetv2_predict", "-c", "{config}", "-d", "{dataset}", "-f", {folds}, "--save_probabilities"]
CMD ["-i", "/data/in", "-o", "/data/out"]
"""
