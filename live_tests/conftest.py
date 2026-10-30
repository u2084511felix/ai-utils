"""Configure explicit paid OpenAI checks."""

import os

import pytest
import pytest_asyncio
from openai import AsyncOpenAI, OpenAI

from aiutils import make_retry_policy, make_budget, make_client
from aiutils.generate import make_generate


def pytest_addoption(parser):
    parser.addoption("--openai-live", action="store_true", help="Permit paid OpenAI test calls.")


@pytest.fixture
def live_client(request):
    if not request.config.getoption("--openai-live"):
        pytest.skip("Use --openai-live to permit paid OpenAI calls.")
    if not os.getenv("OPENAI_API_KEY"):
        raise pytest.UsageError("OPENAI_API_KEY is required for paid checks.")
    sync = OpenAI(max_retries=0, timeout=30)
    asynchronous = AsyncOpenAI(max_retries=0, timeout=30)
    client = make_client(client=sync, async_client=asynchronous,
                         retry_policy=make_retry_policy(retries=0), budget=make_budget(1))
    yield client
    sync.close()


@pytest_asyncio.fixture
async def async_live(live_client):
    yield live_client
    await live_client.async_client.close()


@pytest.fixture
def live_model():
    model = os.getenv("OPENAI_LIVE_MODEL")
    if not model:
        raise pytest.UsageError("OPENAI_LIVE_MODEL is required for paid checks.")
    return model


@pytest.fixture
def chat_params(live_model):
    return dict(model=live_model, messages=[{"role": "user", "content": "Reply with OK."}],
                max_completion_tokens=256)


@pytest.fixture
def structured_gen(async_live):
    return make_generate(client=async_live, max_completion_tokens=256)
