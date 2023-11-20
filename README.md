# Minimal annotation muscle segmentation (museg)

[![CI](https://github.com/fabianbalsiger/minimal-annotation-muscle-segmentation/actions/workflows/main.yaml/badge.svg)](https://github.com/fabianbalsiger/minimal-annotation-muscle-segmentation/actions/workflows/main.yaml)

Minimal annotation muscle segmentation (museg).

# Usage



# Development

A little project cheatsheet:

  - **pre-commit:** `pre-commit run --all-files --hook-stage=manual`
  - **pytest:** `pytest` or `pytest -s`
  - **coverage:** `coverage run -m pytest` followed by `coverage html` or `coverage report`
  - **poetry sync:** `poetry install --no-root --sync`
  - **updating requirements:** see [docs/updating_requirements.md](docs/updating_requirements.md)
  - **create towncrier entry:** `towncrier create 123.added --edit`


Follow these steps for the Initial project setup as well as setting up a new development environment:

1. See [docs/getting_started.md](docs/getting_started.md) or [docs/quickstart.md](docs/quickstart.md)
   for how to get up & running.
2. Check [docs/project_specific_setup.md](docs/project_specific_setup.md) for project specific setup.
3. See [docs/using_poetry.md](docs/using_poetry.md) for how to update Python requirements using
   [Poetry](https://python-poetry.org/).
4. See [docs/detect_secrets.md](docs/detect_secrets.md) for more on creating a `.secrets.baseline`
   file using [detect-secrets](https://github.com/Yelp/detect-secrets).
5. See [docs/using_towncrier.md](docs/using_towncrier.md) for how to update the `CHANGELOG.md`
   using [towncrier](https://github.com/twisted/towncrier).
