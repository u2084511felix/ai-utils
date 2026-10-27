import pickle

import pytest

from aiutils.models import ClassificationModels, GoogleModels, OpenAIModels, TextModels, Vendor


@pytest.mark.parametrize("enum,value", [(GoogleModels, "gemini-2.5-pro"), (OpenAIModels, "gpt-4o"), (TextModels, "gpt-5.2"), (TextModels, "gemini-3-pro-preview"), (ClassificationModels, "jev-latest"), (Vendor, "openai")])
def test_models_are_string_enums(enum, value):
    """GIVEN a model name.
    WHEN the model member is created.
    THEN the member has the expected text value and can be restored from a saved value."""
    member = enum(value)
    result = str(member)
    actual = (result, member, member in list(enum), pickle.loads(pickle.dumps(member)) is member)
    expected = (value, value, True, True)
    assert actual == expected


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
