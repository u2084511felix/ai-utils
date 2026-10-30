"""Check schema and pricing contracts through public operations."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from jsonschema import SchemaError
from pydantic import BaseModel

from aiutils.budget import TokenUsage, make_budget
from aiutils.generate import make_generate
from aiutils.pricing import calculate_token_cost, convert_currency
from aiutils.schemas import validate_schema


@pytest.fixture
def wrapper():
    return {"name": "Result", "strict": True, "schema": {
        "type": "object", "properties": {"value": {"type": "string"}},
        "required": ["value"], "additionalProperties": False,
    }}


@pytest.fixture
def schema_case(wrapper, form, dialect):
    schema = deepcopy(wrapper)
    if dialect is not None:
        schema["schema"]["$schema"] = dialect
    return json.dumps(schema) if form == "string" else schema, schema


@pytest.mark.parametrize("form", ["string", "dict"])
@pytest.mark.parametrize("dialect", [None, "http://json-schema.org/draft-07/schema#", "https://json-schema.org/draft/2020-12/schema"])
def test_named_wrapper_is_valid(schema_case, form, dialect):
    """GIVEN a valid named wrapper. WHEN it is checked. THEN its schema is retained."""
    supplied, expected = schema_case
    result = validate_schema(supplied)
    assert result == expected


@pytest.fixture
def invalid_options():
    return dict(system_msg="system", prompt="question", input_tokens=1)


@pytest.fixture
def contract_gen(client):
    client.budget = make_budget()
    return make_generate(client=client)


@pytest.fixture
def invalid_schema(schema):
    return schema


_INVALID = [
    ("not json", json.JSONDecodeError), ("[]", ValueError), ([], TypeError),
    (1, TypeError), (None, TypeError), ({"type": "object"}, ValueError),
    ({"name": "", "schema": {}}, ValueError), ({"name": " ", "schema": {}}, ValueError),
    ({"name": 1, "schema": {}}, ValueError), ({"name": "Result", "schema": []}, ValueError),
    ({"name": "Result", "schema": {"type": "wrong"}}, SchemaError),
    ({"name": "Result", "schema": {"required": True}}, SchemaError),
    ({"name": "Result", "schema": {}, "strict": "true"}, ValueError),
    ({"name": "Result", "schema": {}, "description": 1}, ValueError),
    ({"name": "Result", "schema": {}, "extra": True}, ValueError),
    ({"name": "Result", "schema": {"$schema": "https://example.com/schema"}}, ValueError),
    ({"name": "Result", "schema": {"minimum": float("nan")}}, ValueError),
]


@pytest.mark.parametrize("schema,error", _INVALID)
@pytest.mark.asyncio
async def test_invalid_schema_leaves_history_unchanged(contract_gen, invalid_options, schema, error):
    """GIVEN an invalid schema. WHEN output is requested. THEN history stays empty."""
    with pytest.raises(error):
        await contract_gen.structured_output(schema=schema, **invalid_options)
    expected = []
    assert contract_gen.messages == expected


@pytest.mark.parametrize("schema,error", _INVALID)
@pytest.mark.asyncio
async def test_invalid_schema_adds_no_reservation(contract_gen, invalid_options, schema, error):
    """GIVEN an invalid schema. WHEN output is requested. THEN estimated spend stays zero."""
    with pytest.raises(error):
        await contract_gen.structured_output(schema=schema, **invalid_options)
    expected = 0
    assert contract_gen.client.budget.estimated_usd == expected


@pytest.mark.parametrize("schema,error", _INVALID)
@pytest.mark.asyncio
async def test_invalid_schema_makes_no_transport_call(contract_gen, invalid_options, schema, error):
    """GIVEN an invalid schema. WHEN output is requested. THEN transport is not called."""
    with pytest.raises(error):
        await contract_gen.structured_output(schema=schema, **invalid_options)
    expected = 0
    assert contract_gen.client.async_client.chat.completions.create.call_count == expected


@pytest.mark.parametrize("schema", [BaseModel, object, int, {}, "Result"])
@pytest.mark.asyncio
async def test_invalid_pydantic_model_leaves_history_unchanged(contract_gen, invalid_options, schema):
    """GIVEN an invalid model. WHEN parsed output is requested. THEN history stays empty."""
    with pytest.raises(TypeError):
        await contract_gen.structured_output(schema=schema, input_type="pydantic", **invalid_options)
    expected = []
    assert contract_gen.messages == expected


class Result(BaseModel):
    value: str


@pytest.fixture
def parsed_gen(contract_gen, response_factory):
    response = response_factory(content='{"value":"hello"}')
    response.choices[0].message.parsed = Result(value="hello")
    contract_gen.client.async_client.chat.completions.parse = AsyncMock(return_value=response)
    return contract_gen


@pytest.mark.asyncio
async def test_pydantic_output_retains_parsed_model(parsed_gen, invalid_options):
    """GIVEN a supported model. WHEN parsed output is requested. THEN the model is retained."""
    reply = await parsed_gen.structured_output(schema=Result, input_type="pydantic", **invalid_options)
    expected = Result(value="hello")
    assert reply.parsed == expected


@pytest.mark.asyncio
async def test_pydantic_output_retains_reported_usage(parsed_gen, invalid_options):
    """GIVEN a supported model. WHEN parsed output is requested. THEN usage is retained."""
    reply = await parsed_gen.structured_output(schema=Result, input_type="pydantic", **invalid_options)
    expected = TokenUsage(input_tokens=10, output_tokens=5, reasoning_tokens=2)
    assert reply.usage == expected


@pytest.mark.parametrize("inputs,outputs,cached", [(-1, 0, 0), (True, 0, 0), (0, 1.5, 0), (1, 0, 2), (float("inf"), 0, 0)])
def test_pricing_rejects_invalid_counts(inputs, outputs, cached):
    """GIVEN invalid counts. WHEN cost is calculated. THEN validation fails."""
    with pytest.raises(ValueError) as error:
        calculate_token_cost("gpt-5.2", inputs, outputs, cached)
    expected = ValueError
    assert error.type is expected


def test_cached_input_uses_total_input_convention():
    """GIVEN half cached input. WHEN cost is calculated. THEN cached input is charged once."""
    result = calculate_token_cost("gpt-5.2", 1_000_000, cached_input_tokens=500_000)
    expected = 0.9625
    assert result == pytest.approx(expected)


def test_cached_input_requires_defined_rate():
    """GIVEN no cached price. WHEN cached usage is priced. THEN validation fails."""
    with pytest.raises(ValueError) as error:
        calculate_token_cost("gpt-5.2-pro", 10, cached_input_tokens=1)
    expected = ValueError
    assert error.type is expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_conversion_rejects_nonfinite_amounts(value):
    """GIVEN a nonfinite amount. WHEN conversion is requested. THEN validation fails."""
    with pytest.raises(ValueError) as error:
        convert_currency(value, currency="USD")
    expected = ValueError
    assert error.type is expected


@pytest.fixture
def default_factory(client, monkeypatch):
    from aiutils import generate
    monkeypatch.setattr(generate, "make_client", lambda: client)
    return client


def test_generation_factory_creates_default_client(default_factory):
    """GIVEN no injected client. WHEN generation is created. THEN the factory supplies it."""
    result = make_generate()
    assert result.client is not None


@pytest.fixture
def cached_response(client, response_factory, shape):
    from types import SimpleNamespace as NS
    response = response_factory()
    if shape == "responses":
        response = NS(output=[], output_text="hello", usage=NS(input_tokens=10, output_tokens=5,
                      input_tokens_details=NS(cached_tokens=4)))
    else:
        response.usage.prompt_tokens_details = NS(cached_tokens=4)
    client.client.responses.create.return_value = response
    return client


@pytest.mark.parametrize("shape", ["responses", "chat"])
def test_cached_usage_is_retained(cached_response, shape):
    """GIVEN cached usage in either response shape. WHEN a reply is read. THEN it is retained."""
    reply = cached_response.responses_create("question", model="gpt-5.2")
    expected = 4
    assert reply.usage.cached_input_tokens == expected


@pytest.mark.parametrize("schema", [None, {}, "Result", int])
@pytest.mark.asyncio
async def test_direct_parsing_rejects_invalid_model(contract_gen, schema):
    """GIVEN invalid parsing input. WHEN the client is called. THEN no reservation is added."""
    with pytest.raises(TypeError):
        await contract_gen.client.async_chat_completions_body({"response_format": schema}, input_type="pydantic", input_tokens=1)
    expected = 0
    assert contract_gen.client.budget.spent_usd == expected


@pytest.fixture
def schema_helper(client, monkeypatch, response_factory):
    from types import SimpleNamespace as NS
    from aiutils.modules import structured_output
    from aiutils.modules.function_call import FunctionCall
    expected = '{"name":"Result","schema":{"type":"object"}}'
    tool = NS(function=NS(arguments=expected))
    client.async_client.chat.completions.create.return_value = response_factory(content=None, tool_calls=[tool])
    monkeypatch.setattr(structured_output, "FunctionCall", lambda: FunctionCall(client=client))
    return structured_output, expected


@pytest.mark.asyncio
async def test_schema_generation_uses_chat_tool_content(schema_helper):
    """GIVEN a chat tool response. WHEN a schema is generated. THEN tool arguments are read."""
    module, expected = schema_helper
    result = await module.generate_structured_output_schema({"value": "string"})
    assert result == expected


@pytest.fixture
def missing_embedding(client):
    del client.async_client.embeddings.create.return_value.usage
    return client


@pytest.mark.asyncio
async def test_embedding_requires_reported_usage(missing_embedding):
    """GIVEN missing usage. WHEN an embedding is requested. THEN the response is rejected."""
    with pytest.raises(RuntimeError) as error:
        await missing_embedding.make_embeddings("text", "text-embedding-3-small", "float", 1536)
    expected = RuntimeError
    assert error.type is expected


@pytest.fixture
def tool_budget(client):
    from aiutils.modules.function_call import FunctionCall
    client.budget = make_budget()
    return FunctionCall(client=client)


@pytest.mark.asyncio
async def test_tool_generation_uses_shared_budget(tool_budget):
    """GIVEN a shared budget. WHEN tool generation runs. THEN reported usage is charged."""
    await tool_budget.function_call_legacy_structured_output("system", "prompt", model="gpt-5.2", input_tokens=1)
    expected = 0.0000875
    assert tool_budget.client.budget.confirmed_usd == pytest.approx(expected)
