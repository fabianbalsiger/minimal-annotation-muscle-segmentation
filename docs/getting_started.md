# Getting Started

This document contains instructions to get a fully working development environment on a Linux machine.
The instructions follow the [hypermodern Python series](https://medium.com/@cjolowicz/hypermodern-python-d44485d9d769#6e8a).


## 1. [pyenv](https://github.com/pyenv/pyenv)

pyenv is a Python version manager and contains a plugin [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) that manages virtualenvs across multiple Python versions. Learn more about it [here](https://realpython.com/intro-to-pyenv/). Install it by:

```sh
curl https://pyenv.run | bash
```

Configure by adding the following to your `~/.bashrc` or equivalent:

```sh
# Pyenv environment variables
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Basic usage:

```sh
# Check Python versions
pyenv install --list

# See installed Python versions
pyenv versions

# Change to the Python version you just installed
pyenv shell <version>

# Show list of current environments
pyenv virtualenvs

# Activate and deactivate environment
pyenv activate <name>
pyenv deactivate
```

## 2. Get the repository and initialise the environment

⚠️ N.B. You should replace `REPO_GIT_URL` here with your actual URL to your GitHub repo.

```sh
git clone ${REPO_GIT_URL}
pyenv shell $(sed "s/\/envs.*//" .python-version)

# If necessary, install required prerequisites first: https://github.com/pyenv/pyenv/wiki/Common-build-problems
# Install the Python version
pyenv install $(sed "s/\/envs.*//" .python-version)

# Create a virtualenv
pyenv virtualenv $(sed "s/\/envs\// /" .python-version)
# The venv will be located at $(pyenv root)/versions

pyenv activate $(cat .python-version)
python -V  # check this is the correct version of Python
python -m pip install --upgrade pip
```

Note that the environment should load and unload automatically in the directory. Check it's working by cd-ing into & out of the repo.


## 3. Install Python requirements into the virtual environment using [Poetry](https://python-poetry.org/docs/)

Install Poetry onto your system by following the instructions here: [https://python-poetry.org/docs/]

Note that Poetry "lives" outside of project/environment, and if you follow the recommended install
process it will be installed isolated from the rest of your system.

```sh
# Add the poetry plugin "up" for upgrading dependencies: https://github.com/MousaZeidBaker/poetry-plugin-up
poetry self add poetry-plugin-up

# Update Poetry regularly as you would any other system-level tool. Poetry is environment agnostic,
# it doesn't matter if you run this command inside/outside the virtualenv.
poetry self update

# This command should be run inside the virtualenv.
poetry install --no-root --sync
```


## 4. Add secrets into .env

  - Run `cp .env.template .env` and update the secrets.
  - [Install direnv](https://direnv.net/) to autoload environment variables specified in `.env`
  - Run `direnv allow` to authorise direnv to load the secrets from `.env` (by reading `.envrc`) into the environment
    (these will unload when you `cd` out of the repo; note that you will need to re-run this
    command whenever you change `.env`)


## 5. Initialise the `detect-secrets` pre-commit hook

We use [`detect-secrets`](https://github.com/Yelp/detect-secrets) to check that no secrets are
accidentally committed. Skip this step if the project has already been initialized.
Please read [docs/detect_secrets.md](docs/detect_secrets.md) for more information.


```shell
# Generate a baseline
detect-secrets scan > .secrets.baseline

# You may want to check/amend the exclusions in `.pre-commit-config.yaml` e.g.
detect-secrets --verbose scan \
    --exclude-files 'poetry\.lock' \
    --exclude-files '\.secrets\.baseline' \
    --exclude-files '\.env\.template' \
    --exclude-files '.*\.ipynb$', \
    --exclude-secrets 'password|ENTER_PASSWORD_HERE|INSERT_API_KEY_HERE' \
    --exclude-lines 'integrity=*sha' \
    > .secrets.baseline

# Audit the generated baseline
detect-secrets audit .secrets.baseline
```

When you run this command, you'll enter an interactive console. This will present you with a list
of high-entropy string and/or anything which could be a secret. It will then ask you to verify
whether this is the case. This allows the hook to remember false positives in the future, and alert
you to new secrets.


## 6. Project-specific setup

Please check [docs/project_specific_setup.md](docs/project_specific_setup.md) for further instructions.
