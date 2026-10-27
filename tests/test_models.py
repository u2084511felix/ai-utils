import pickle

import pytest

from aiutils.models import ClassificationModels, GoogleModels, OpenAIModels, TextModels, Vendor


@pytest.mark.parametrize("enum,value", [
    (GoogleModels, "gemini-2.5-pro"), (OpenAIModels, "gpt-4o"),
    (TextModels, "gpt-5.2"), (TextModels, "gemini-3-pro-preview"),
    (ClassificationModels, "jev-latest"), (Vendor, "openai"),
])
def test_models_are_string_enums(enum, value):
    member = enum(value)
    assert str(member) == value
    assert member == value
    assert member in list(enum)
    assert pickle.loads(pickle.dumps(member)) is member


def test_legacy_aliases_remain_available():
    assert TextModels.hipster is TextModels.gpt_4o
    assert TextModels.o1preview == "o1-preview"


@pytest.mark.parametrize("enum", [TextModels, GoogleModels, OpenAIModels, ClassificationModels])
def test_unknown_model_rejected(enum):
    with pytest.raises(ValueError):
        enum("not-a-model")
