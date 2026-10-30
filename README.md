# ai-utils

Utilities for AI clients, generation, embeddings, shared budgets, and file patches.
Use Python 3.13 or newer.

## Install and test

```sh
poetry install --with dev
poetry run pytest
```

The normal suite runs without API credentials or network access. It uses fake
clients and mock HTTP transports. Tokenizer tests use local token data. File
operations use temporary directories. No paid checks run in this suite.

To measure statements and branches in `aiutils`:

```sh
poetry run pytest --cov=aiutils --cov-report=term-missing
```

## Generation with a shared budget

Set `OPENAI_API_KEY` before you create an OpenAI client. Imports do not create
clients or require API keys. Use this example inside an async function:

```python
from aiutils import make_retry_policy, make_budget, make_client
from aiutils.generate import make_generate

budget = make_budget(
    2.0,
    token_limits={"gpt-5.2": 60_000},
    window_seconds=60,
)
client = make_client(budget=budget, retry_policy=make_retry_policy(retries=2))
generate = make_generate(client=client, max_completion_tokens=512)
reply = await generate.generate(
    system_msg="Give a short answer.",
    prompt="What is a token?",
    input_tokens=128,
)
print(reply.content)
print(budget.confirmed_usd, budget.unresolved_usd)
```

Supply an `input_tokens` estimate when a client has a budget. Include the full
request context in that estimate. All calls that use the same budget share its
capacity and spend limit. A separate client can use the same budget. A classifier
can use the public reservation operations directly.

An input estimate does not guarantee a strict final spend cap. A call can use
more tokens than its estimate. See [the migration contract](MIGRATION.md) for
capacity windows, reservations, failure billing, and cancellation.

## Structured output

Use a named JSON Schema wrapper as a dictionary or JSON string:

```python
schema = {
    "name": "Answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}
reply = await generate.structured_output(
    system_msg="Give a short answer.",
    prompt="What is a token?",
    schema=schema,
    input_tokens=128,
)
```

Validation uses JSON Schema Draft 2020-12 by default. An explicit Draft 7
`$schema` is also supported. A valid generic schema can still violate a provider
restriction. Generic validation does not establish provider acceptance.

For Pydantic output, supply a model class and `input_type="pydantic"`. This path
uses the SDK parsing API. The reply retains text, reported usage, the provider
response, and the parsed model in `reply.parsed`.

## Paid OpenAI checks

These checks are separate from the normal suite and pre-commit. They make paid
requests. They were created for separate manual use.

Set real credentials and select a priced model that supports Chat Completions,
Responses, structured output, and function tools:

```sh
export OPENAI_API_KEY='your-key'
export OPENAI_LIVE_MODEL='gpt-5.2'
poetry run pytest live_tests --openai-live
```

The suite disables retries and sets an input spend limit of USD 1 per test.
The limit applies to input reservations. It does not guarantee a strict final
charge limit. The suite checks sync chat, async chat, Responses, Pydantic output,
and tool-call usage. No live call runs without the explicit flag.

## Pre-commit

```sh
poetry run pre-commit install
poetry run pre-commit run --all-files
```

Each commit checks TOML and YAML syntax, checks merge conflict markers, and runs
the normal pytest suite. A failure blocks the commit. The first hook run downloads
the pinned validation hooks. The pytest suite itself runs offline.

The [OpenAI Python SDK reference](https://developers.openai.com/api/reference/python)
describes SDK retry controls. The
[structured-output guide](https://developers.openai.com/api/docs/guides/structured-outputs)
describes the parsing API and provider schema restrictions.
