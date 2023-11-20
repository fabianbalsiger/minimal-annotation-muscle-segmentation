# Quickstart

This document contains instructions _only_ to get a fully working development environment for
running this repo. For pre-requisites (e.g. `pyenv` install instructions) plus details on what's
being installed and why, please see [docs/getting_started.md](docs/getting_started.md).

We assume the following are installed and configured:
  - [pyenv](https://github.com/pyenv/pyenv)
  - [Poetry](https://python-poetry.org/docs/)
  - [direnv](https://direnv.net/)


## Part 1: Generic Python setup

```sh
# Get the repo
git clone ${REPO_GIT_URL}

# Install Python
pyenv install $(sed "s/\/envs.*//" .python-version)

# Setup the virtualenv
pyenv virtualenv $(sed "s/\/envs\// /" .python-version)
python -V
python -m pip install --upgrade pip

# Install dependencies with Poetry
poetry self update
poetry install --no-root --sync

# Edit .env for storing secrets (.env is a copy of .env.template)
direnv allow

# Create and audit secrets baseline
# N.B. Adjust the exclusions here depending on your needs (check .pre-commit-config.yaml)
detect-secrets --verbose scan \
    --exclude-files 'poetry\.lock' \
    --exclude-files '\.secrets\.baseline' \
    --exclude-files '\.env\.template' \
    --exclude-secrets 'password|ENTER_PASSWORD_HERE|INSERT_API_KEY_HERE' \
    --exclude-lines 'integrity=*sha' \
    > .secrets.baseline

detect-secrets audit .secrets.baseline
```


## Part 2: Project-specific setup

Please check [docs/project_specific_setup.md](docs/project_specific_setup.md) for further instructions.
