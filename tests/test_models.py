import pickle

import pytest

from aiutils.models import ClassificationModels, GoogleModels, OpenAIModels, TextModels, Vendor


@pytest.mark.parametrize("enum,value", [(GoogleModels, "gemini-2.5-pro"), (OpenAIModels, "gpt-4o"), (TextModels, "gpt-5.2"), (TextModels, "gemini-3-pro-preview"), (ClassificationModels, "jev-latest"), (Vendor, "openai")])
def test_model_member_has_text(enum, value):
    """GIVEN a model name.
    WHEN the model member is created.
    THEN model member has text."""
    member = enum(value)
    result = str(member)
    expected = value
    assert result == expected


@pytest.mark.parametrize("enum,value", [(GoogleModels, "gemini-2.5-pro"), (OpenAIModels, "gpt-4o"), (TextModels, "gpt-5.2"), (TextModels, "gemini-3-pro-preview"), (ClassificationModels, "jev-latest"), (Vendor, "openai")])
def test_model_member_equals_value(enum, value):
    """GIVEN a model name.
    WHEN the model member is created.
    THEN model member equals value."""
    member = enum(value)
    expected = value
    assert member == expected


@pytest.mark.parametrize("enum,value", [(GoogleModels, "gemini-2.5-pro"), (OpenAIModels, "gpt-4o"), (TextModels, "gpt-5.2"), (TextModels, "gemini-3-pro-preview"), (ClassificationModels, "jev-latest"), (Vendor, "openai")])
def test_model_member_belongs_to_enum(enum, value):
    """GIVEN a model name.
    WHEN the model member is created.
    THEN model member belongs to enum."""
    member = enum(value)
    expected = member
    assert expected in list(enum)


@pytest.mark.parametrize("enum,value", [(GoogleModels, "gemini-2.5-pro"), (OpenAIModels, "gpt-4o"), (TextModels, "gpt-5.2"), (TextModels, "gemini-3-pro-preview"), (ClassificationModels, "jev-latest"), (Vendor, "openai")])
def test_model_member_survives_serialization(enum, value):
    """GIVEN a model name.
    WHEN the model member is created.
    THEN model member survives serialization."""
    member = enum(value)
    expected = member
    assert pickle.loads(pickle.dumps(member)) is expected


@pytest.mark.parametrize("alias,expected", [(TextModels.hipster, TextModels.gpt_4o), (TextModels.o1preview, TextModels.o1_preview)])
def test_legacy_aliases_remain_available(alias, expected):
    """GIVEN a legacy model name.
    WHEN the model member is created.
    THEN the result is the matching model member."""
    result = TextModels(alias)
    assert result is expected


@pytest.mark.parametrize("enum", [TextModels, GoogleModels, OpenAIModels, ClassificationModels])
def test_unknown_model_rejected(enum):
    """GIVEN an unknown model name.
    WHEN the model member is created.
    THEN the unknown model is rejected."""
    with pytest.raises(ValueError) as exc:
        enum("not-a-model")
    expected = ValueError
    assert exc.type == expected
