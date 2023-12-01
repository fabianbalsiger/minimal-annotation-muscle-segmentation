# Minimal annotation muscle segmentation (MuSeg)

[![CI](https://github.com/fabianbalsiger/minimal-annotation-muscle-segmentation/actions/workflows/main.yaml/badge.svg)](https://github.com/fabianbalsiger/minimal-annotation-muscle-segmentation/actions/workflows/main.yaml)

Minimal annotation muscle segmentation (MuSeg) based on artificial intelligence (AI), short MuSegAI. If you use this work, please consider [citing it](#citation).

The repository provides a command line interface (CLI) based on a Docker image for segmentation. Prerequisites for the segmentation are two Dixon MRI
volumes at echo times 1.95 and 2.75 ms (in- and out-of-phase volumes).
Currently, a segmentation model for the thigh muscles is available, which segments 13 muscles (see `./docker/labels_thigh.txt`).
The volumes can either contain both thighs or left and right only.

# Installation

Install the MuSeg-AI package by (tested on Ubuntu 22.04 with Python 3.10)

```bash
git clone https://github.com/fabianbalsiger/minimal-annotation-muscle-segmentation.git
cd minimal-annotation-muscle-segmentation
pip install .[cli]
```

You can also directly work with the Docker image available on [DockerHub](https://hub.docker.com/repository/docker/fabianbalsiger/museg/general), pull it by

```bash
docker pull fabianbalsiger/museg:thigh-model3
```

Follow the [docker/README.md](./docker/README.md) instructions on how to execute a Docker container for segmentation.

# CLI usage

Let's assume having volumes saved in a directory `in` as `volume195.mha` and `volume275.mha` (the names can be arbritary). Segment them by

```bash
museg-ai in/volume195.mha in/volume275.mha
```

The files `volume195.mha`, the segmentation, and `labels.txt`, an [ITK-SNAP](https://itksnap.org) labels file, will be saved at the location where the script has been executed.

**NOTE:** If the Docker image for segmentation has not yet been pulled, it will be done automatically, which might take a while.

Print all available options by

```bash
museg-ai --help
```

# Citation

The scientific article is currently under review. Please open an issue if you want to cite it before acceptance.

# Development

A little project cheatsheet for development:

  - **pre-commit:** `pre-commit run --all-files --hook-stage=manual`
  - **pytest:** `pytest` or `pytest -s`
  - **coverage:** `coverage run -m pytest` followed by `coverage html` or `coverage report`
  - **create towncrier entry:** `towncrier create 123.added --edit`


Follow these steps for the initial project setup as well as setting up a new development environment:

1. See [docs/getting_started.md](docs/getting_started.md) or [docs/quickstart.md](docs/quickstart.md)
   for how to get up & running.
2. Check [docs/project_specific_setup.md](docs/project_specific_setup.md) for project specific setup.
3. See [docs/detect_secrets.md](docs/detect_secrets.md) for more on creating a `.secrets.baseline`
   file using [detect-secrets](https://github.com/Yelp/detect-secrets).
4. See [docs/using_towncrier.md](docs/using_towncrier.md) for how to update the `CHANGELOG.md`
   using [towncrier](https://github.com/twisted/towncrier).
