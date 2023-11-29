# Build Docker image

Use the following command to build the Docker image `museg:thigh-model3` (muscle segmentation thigh model version 3):

```bash
docker build --rm -f Dockerfile.thigh -t museg:thigh-model3 .
```

Prerequisites are that the directory contains a copy of the nnU-Net directory `nnUNet_trained_models` with the trained model, which is not part of
this repository due to its size. Further, it is required to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).

# Run the Docker image

To run the muscle segmentation, the volumes to be segmented need to be put into the correct format.
The algorithm requires two volumes per subject named `suffix_0000.nii.gz` for the in-phase volume at echo time 1.95 ms and `suffix_0001.nii.gz`
for the out-of-phase volume at echo time 2.75 ms, where `suffix` can be an arbritary string but must be kept equal for both volume files.
Both volumes are to be put inside the same directory, e.g., named `in`. Insided this directory can also be multiple subjects with different suffixes.

```
.
├── in  # Directory with the input volumes (volumes to be segmented)
│   ├── subject1_0000.nii.gz  # Volume 1 in-phase at echo time 1.95 ms
│   ├── subject1_0001.nii.gz  # Volume 2 out-of-phase at echo time 2.75 ms
│   ├── subject_long_suffix2_0000.nii.gz
│   └── subject_long_suffix2_0001.nii.gz
```

To infer the thigh muscle segmentation execute within the directory that contains `in` and `out`

```bash
docker run --rm --gpus all -v "$PWD":/data museg:thigh-model3
```

The command will run the Docker image `museg:thigh-model3` with access to the GPUs (`--gpus all`) and by mounting the current directory `-v "$PWD":/data`.
It will also remove the contained after it finished (`--rm`).
After successful executing, the segmentations will be saved into a new directory `out`.

```
.
├── in
│   ├── subject_0000.nii.gz
│   └── subject_0001.nii.gz
└── out
    ├── subject.nii.gz
    ├── subject_long_suffix2.nii.gz
    ├── plans.pkl
    ├── postprocessing.json
    └── prediction_time.txt
```

## Overwriting directory names

You can adapt the `in` and `out` directory names as you wish by specifying the `-i` and `-o` parameters.
Let's assume you put the data inside `mycustomin` and want the segmentation to be saved inside `mycustomout`, execute the Docker by

```bash
docker run --rm --gpus all -v "$PWD":/data museg:thigh3 -i ./data/mycustomin -o ./data/mycustomout
```

## Modifying nnU-Net inference parameters

The entry point of the Docker image is the nnU-Net's `nnUNet_predict`. You can modify any parameters of this function by passing it to the command.
For instance, use `--disable_tta` to speed up inference time by approximately factor 8 by disabling test-time augmentation:

```bash
docker run --rm --gpus all -v "$PWD":/data museg:thigh-model3 --disable_tta
```

Use `docker run --rm --gpus all -v "$PWD":/data museg:thigh-model3 --help` to learn more about inference parameters.


## Inference without a GPU

If no GPU is available, just drop the `--gpus all`

```bash
docker run --rm -v "$PWD":/data museg:thigh-model3
```
