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

The suite runs offline without API credentials. Fixtures provide placeholder
environment variables for import-time SDK initialization, inject mocked clients,
and reject socket connections. Tokenizer tests use a local fake so they do not
download vocabulary files. File operations run in pytest temporary directories;
shell commands and code generation are mocked.

Tests cover model lookup, pricing and conversion, budget validation and concurrent
reservations, client dispatch and error handling, generation and structured
requests, embeddings, patch application and undo, prompt rendering, schema helpers,
and audio requests.

Known defects are documented with `xfail(strict=True)` tests. Pytest prints their
reasons in its summary; a fixed defect becomes an unexpected pass and fails the
suite until its marker is removed. These cases include token-cost arithmetic,
price conversion, empty provider choices, Responses API parsing, stale reasoning
settings, token counting, the default prompt template path, speech presets, and
WebSearch construction. An expected failure does not indicate working behavior.
