# Updating requirements

```sh
# Create feature branch
poetry install --no-root --sync
poetry up --latest
# If any issues, likely due to two "latest" packages conflicting. See example below.
poetry install --no-root --sync
pre-commit run --all-files --hook-stage=manual
poetry install --no-root --sync
pytest
# Commit & push
```

If the latest version of `green` requires `blue (>=1.2, <1.3)` and the latest version of `blue` is
`1.4` then you will encounter a `SolverProblemError`, for example:

```sh
SolverProblemError

Because green (0.8) depends on blue (>=1.2,<1.3)
and no versions of green match (>=0.8,<1.0) requires blue (>=1.2,<1.3).
So, because src depends on both blue (^1.4) and green (^0.8), version solving failed.
```

In this situation, do the following:

    - Comment out `blue`
    - Re-run `poetry up --latest`
    - Handle any other new package conflicts the same way until poetry up resolves
    - Uncomment out `blue` with package version that works with `green`, e.g. `blue = "^1.2"`
    - Run `poetry-regenerate` onwards
