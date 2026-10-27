from pydantic import ValidationError
import pytest

from aiutils import pricing
from aiutils.pricing import Currency, Price, calculate_token_cost, convert, get_price


@pytest.mark.parametrize("model", ["gpt-4o", " GPT-4O ", "GpT-4o"])
def test_price_lookup_normalizes_name(model):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN price lookup normalizes name."""
    price = get_price(model)
    expected = "gpt-4o"
    assert price.name == expected


@pytest.mark.parametrize("model", ["gpt-4o", " GPT-4O ", "GpT-4o"])
def test_price_lookup_returns_rate(model):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN price lookup returns rate."""
    price = get_price(model)
    expected = 2.5
    assert price.input == expected


def test_unknown_price_has_helpful_error():
    """GIVEN a model with no listed price.
    WHEN the price operation is called.
    THEN the error identifies the unknown model."""
    with pytest.raises(NotImplementedError, match="no-such-model") as exc:
        get_price("no-such-model")
    expected = NotImplementedError
    assert exc.type == expected


@pytest.fixture
def custom_price():
    price = Price(name="test", input=1, output=2, cached_input=None)
    return price


def test_custom_price_table(custom_price):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN the result is the price from the supplied table."""
    price = custom_price
    result = get_price(" TEST ", {"test": price}) is price
    expected = True
    assert result == expected


@pytest.fixture
def invalid_price(field, value):
    values = dict(name="test", input=1, output=2, cached_input=None)
    values[field] = value
    return values


@pytest.mark.parametrize("field,value", [("input", -1), ("output", -1), ("cached_input", -1), ("discount", 0), ("name", "")])
def test_invalid_price_rejected(invalid_price, field, value):
    """GIVEN an invalid price field.
    WHEN the price operation is called.
    THEN the invalid price field is rejected."""
    values = invalid_price
    with pytest.raises(ValidationError) as exc:
        Price(**values)
    expected = ValidationError
    assert exc.type == expected


@pytest.mark.parametrize("currency,rate", [(Currency.USD, 1), (Currency.GBP, 2), (Currency.EUR, 4)])
def test_convert_uses_supplied_exchange_rate(currency, rate):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN the amount uses the supplied exchange rate."""
    result = convert(100, currency=currency, rates={currency: rate})
    expected = 100 / rate
    assert result == expected


@pytest.mark.parametrize("amount", [0, -10, 1.25])
def test_usd_conversion_is_identity(amount):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN the amount does not change."""
    result = convert(amount, currency="USD")
    expected = amount
    assert result == expected


def test_conversion_rejects_invalid_currency():
    """GIVEN an unknown currency.
    WHEN the price operation is called.
    THEN the currency is rejected."""
    with pytest.raises(ValueError) as exc:
        convert(1, currency="XYZ")
    expected = ValueError
    assert exc.type == expected


@pytest.mark.parametrize("rates,error,message", [({}, KeyError, "not available"), ({"EUR": 0}, ZeroDivisionError, "zero")])
def test_invalid_exchange_rates(rates, error, message):
    """GIVEN a missing or zero exchange rate.
    WHEN the price operation is called.
    THEN the missing or zero rate is rejected."""
    with pytest.raises(error, match=message) as exc:
        convert(1, currency="EUR", rates=rates)
    expected = error
    assert exc.type == expected


def test_zero_tokens_cost_nothing():
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN the cost is zero."""
    result = calculate_token_cost("gpt-4o", 0)
    expected = 0
    assert result == expected


@pytest.mark.parametrize("inputs,outputs,cached,expected", [(1000000, 0, 0, 2.5), (0, 1000000, 0, 10), (0, 0, 1000000, 1.25), (1000000, 1000000, 1000000, 13.75)])
def test_cost_matches_listed_rates(inputs, outputs, cached, expected):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN the cost matches the published token prices."""
    result = calculate_token_cost("gpt-4o", inputs, outputs, cached)
    expected = pytest.approx(expected)
    assert result == expected


@pytest.fixture
def discounted_price():
    price = Price(name="test", input=10, output=20, cached_input=None, discount=2)
    return price


def test_price_conversion_converts_input(discounted_price):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN price conversion converts input."""
    price = discounted_price
    converted = price.convert(currency="GBP", fx_rates={"GBP": 2})
    expected = 5
    assert converted.input == expected


def test_price_conversion_converts_output(discounted_price):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN price conversion converts output."""
    price = discounted_price
    converted = price.convert(currency="GBP", fx_rates={"GBP": 2})
    expected = 10
    assert converted.output == expected


def test_price_conversion_retains_absent_cache(discounted_price):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN price conversion retains absent cache."""
    price = discounted_price
    converted = price.convert(currency="GBP", fx_rates={"GBP": 2})
    expected = None
    assert converted.cached_input is expected


def test_price_conversion_retains_discount(discounted_price):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN price conversion retains discount."""
    price = discounted_price
    converted = price.convert(currency="GBP", fx_rates={"GBP": 2})
    expected = 2
    assert converted.discount == expected


@pytest.fixture
def discount_rates(monkeypatch):
    monkeypatch.setattr(pricing, "get_price", lambda model: Price(name=model, input=4, output=8, cached_input=None, discount=2))
    monkeypatch.setitem(pricing.FX_RATES, Currency.GBP, 2)


def test_cost_applies_discount_and_currency(discount_rates):
    """GIVEN specified prices and currency values.
    WHEN the price operation is called.
    THEN the cost includes the discount and currency conversion."""
    result = calculate_token_cost("test", 1000000, currency="GBP")
    expected = pytest.approx(1)
    assert result == expected


@pytest.fixture
def currency_price():
    price = Price(name="test", input=10, output=20, cached_input=None)
    return price


def test_price_conversion_rejects_zero_exchange_rate(currency_price):
    """GIVEN a zero exchange rate.
    WHEN the price operation is called.
    THEN the zero rate is rejected."""
    price = currency_price
    with pytest.raises(ZeroDivisionError, match="cannot be zero") as exc:
        price.convert(currency="EUR", fx_rates={"EUR": 0})
    expected = ZeroDivisionError
    assert exc.type == expected
