# ai-utils

Utilities for AI clients, generation, embeddings, spend budgets, and file patches.

## Tests

Use Python 3.13 or newer and install the development dependencies with Poetry:

```sh
poetry install --with dev
poetry run pytest
```

To include a coverage report:

```sh
poetry run pytest --cov=aiutils --cov-report=term-missing
```

Coverage configuration measures statements and branches in `aiutils`, excluding
test code. The run refreshes `.coverage`; inspect the saved results with
`poetry run coverage report -m`.

The suite runs offline without API credentials. Fixtures provide placeholder
environment variables for import-time SDK initialization, inject mocked clients,
and reject socket connections. Tokenizer tests use a local fake so they do not
download vocabulary files. File operations run in pytest temporary directories;
shell commands and code generation are mocked.

Tests cover model lookup, pricing and conversion, budget validation and concurrent
reservations, client dispatch and error handling, generation and structured
requests, embeddings, patch application and undo, prompt rendering, schema helpers,
and audio requests.

Regression tests expose known defects as ordinary test failures. These cases
include token-cost arithmetic,
price conversion, empty provider choices, Responses API parsing, stale reasoning
settings, token counting, the default prompt template path, speech presets,
WebSearch construction and request handling, dropped Responses API tools,
structured-output defaults, patch client responses, and failed patch history.

## Pre-commit hooks

After installing the development dependencies, enable the hooks in your checkout:

```sh
poetry run pre-commit install
```

Each commit checks staged TOML and YAML files for valid syntax, checks staged
files for merge conflict markers, and runs the full pytest suite. The test hook
uses the Poetry environment and runs once per commit, without passing staged
filenames to pytest. Any test failure blocks the commit.

To run the hooks manually across all tracked files:

```sh
poetry run pre-commit run --all-files
```

The first run downloads the pinned validation hooks and creates their environment.
The pytest suite itself runs offline.
