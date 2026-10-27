import pytest
from pydantic import ValidationError

from aiutils import pricing
from aiutils.pricing import Currency, Price, calculate_token_cost, convert, get_price


@pytest.mark.parametrize("model", ["gpt-4o", " GPT-4O ", "GpT-4o"])
def test_price_lookup_is_case_insensitive(model):
    assert get_price(model).name == "gpt-4o"
    assert get_price(model).input == 2.5


def test_unknown_price_has_helpful_error():
    with pytest.raises(NotImplementedError, match="no-such-model"):
        get_price("no-such-model")


def test_custom_price_table():
    price = Price(name="test", input=1, output=2, cached_input=None)
    assert get_price(" TEST ", {"test": price}) is price


@pytest.mark.parametrize("field,value", [("input", -1), ("output", -1), ("cached_input", -1), ("discount", 0), ("name", "")])
def test_invalid_price_rejected(field, value):
    values = dict(name="test", input=1, output=2, cached_input=None)
    values[field] = value
    with pytest.raises(ValidationError):
        Price(**values)


@pytest.mark.parametrize("currency,rate", [(Currency.USD, 1), (Currency.GBP, 2), (Currency.EUR, 4)])
def test_convert_uses_supplied_exchange_rate(currency, rate):
    assert convert(100, currency=currency, rates={currency: rate}) == 100 / rate


@pytest.mark.parametrize("amount", [0, -10, 1.25])
def test_usd_conversion_is_identity(amount):
    assert convert(amount, currency="USD") == amount


def test_conversion_rejects_invalid_currency():
    with pytest.raises(ValueError):
        convert(1, currency="XYZ")


def test_missing_and_zero_exchange_rates():
    with pytest.raises(KeyError, match="not available"):
        convert(1, currency="EUR", rates={})
    with pytest.raises(ZeroDivisionError, match="zero"):
        convert(1, currency="EUR", rates={"EUR": 0})


def test_zero_tokens_cost_nothing():
    assert calculate_token_cost("gpt-4o", 0) == 0


@pytest.mark.xfail(strict=True, reason="Token cost divides by the per-million rate and scales twice instead of multiplying once")
@pytest.mark.parametrize("inputs,outputs,cached,expected", [
    (1_000_000, 0, 0, 2.5), (0, 1_000_000, 0, 10),
    (0, 0, 1_000_000, 1.25), (1_000_000, 1_000_000, 1_000_000, 13.75),
])
def test_cost_matches_listed_rates(inputs, outputs, cached, expected):
    assert calculate_token_cost("gpt-4o", inputs, outputs, cached) == pytest.approx(expected)


@pytest.mark.xfail(strict=True, reason="Price.convert incorrectly applies token scaling to per-million price values")
def test_price_conversion_preserves_units_and_discount():
    price = Price(name="test", input=10, output=20, cached_input=None, discount=2)
    converted = price.convert(currency="GBP", fx_rates={"GBP": 2})
    assert converted.input == 5
    assert converted.output == 10
    assert converted.cached_input is None
    assert converted.discount == 2


@pytest.mark.xfail(strict=True, reason="Underlying token-cost arithmetic is incorrect")
def test_cost_applies_discount_and_currency(monkeypatch):
    monkeypatch.setattr(pricing, "get_price", lambda model: Price(name=model, input=4, output=8, cached_input=None, discount=2))
    monkeypatch.setitem(pricing.FX_RATES, Currency.GBP, 2)
    assert calculate_token_cost("test", 1_000_000, currency="GBP") == pytest.approx(1)
