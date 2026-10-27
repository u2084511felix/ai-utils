"""Model token prices and currency conversion."""
from enum import StrEnum


from pydantic import BaseModel, NonNegativeFloat, Field, PositiveFloat


from .models import (
    TextModels as TM, 
    ClassificationModels as CM
)


__PER_MILLION = 1_000_000

class Currency(StrEnum):
    USD = "USD"
    GBP = "GBP"
    EUR = "EUR"

# TODO: Update these rates when currency rates change.
# NOTE: Last updated: 10/26/2026
FX_RATES: dict[Currency, float] = {  
    Currency.USD: 1.0,
    Currency.GBP: 1.33,
    Currency.EUR: 1.12,
}

_PRICES_USD = {  # TODO: Update these prices when provider prices change.
    TM.gpt_5_2: {"input": 1.75, "cached_input": 0.175, "output": 14.0},
    TM.gpt_5_4: {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    TM.gpt_5_4_mini: {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    TM.gpt_5_4_nano: {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    TM.gpt_5_4_pro: {"input": 30.00, "cached_input": None, "output": 180.00},
    TM.gpt_5_2_pro: {"input": 21.00, "cached_input": None, "output": 168.00},
    TM.gpt_5_1: {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    TM.gpt_5: {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    TM.gpt_5_mini: {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    TM.gpt_5_nano: {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    TM.gpt_5_pro: {"input": 15.00, "cached_input": None, "output": 120.00},
    TM.gpt_4_1: {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    TM.gpt_4_1_mini: {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    TM.gpt_4_1_nano: {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    TM.gpt_4o: {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    TM.gpt_4o_mini: {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    TM.o4_mini: {"input": 1.10, "cached_input": 0.275, "output": 4.40},
    TM.o3: {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    TM.o3_mini: {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    TM.o3_pro: {"input": 20.00, "cached_input": None, "output": 80.00},
    TM.o1: {"input": 15.00, "cached_input": 7.50, "output": 60.00},
    TM.o1_mini: {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    TM.o1_pro: {"input": 150.00, "cached_input": None, "output": 600.00},
    TM.gpt_4o_2024_05_13: {"input": 5.00, "cached_input": None, "output": 15.00},
    TM.gpt_4_turbo_2024_04_09: {"input": 10.00, "cached_input": None, "output": 30.00},
    TM.gpt_4_0125_preview: {"input": 10.00, "cached_input": None, "output": 30.00},
    TM.gpt_4_1106_preview: {"input": 10.00, "cached_input": None, "output": 30.00},
    TM.gpt_4_1106_vision_preview: {"input": 10.00, "cached_input": None, "output": 30.00},
    TM.gpt_4_0613: {"input": 30.00, "cached_input": None, "output": 60.00},
    TM.gpt_4_0314: {"input": 30.00, "cached_input": None, "output": 60.00},
    TM.gpt_4_32k: {"input": 60.00, "cached_input": None, "output": 120.00},
    TM.gpt_3_5_turbo: {"input": 0.50, "cached_input": None, "output": 1.50},
    TM.gpt_3_5_turbo_0125: {"input": 0.50, "cached_input": None, "output": 1.50},
    TM.gpt_3_5_turbo_1106: {"input": 1.00, "cached_input": None, "output": 2.00},
    TM.gpt_3_5_turbo_0613: {"input": 1.50, "cached_input": None, "output": 2.00},
    TM.gpt_3_5_0301: {"input": 1.50, "cached_input": None, "output": 2.00},
    TM.gpt_3_5_turbo_instruct: {"input": 1.50, "cached_input": None, "output": 2.00},
    TM.gpt_3_5_turbo_16k_0613: {"input": 3.00, "cached_input": None, "output": 4.00},
    TM.davinci_002: {"input": 2.00, "cached_input": None, "output": 2.00},
    TM.babbage_002: {"input": 0.40, "cached_input": None, "output": 0.40},
    CM.jev_1_14_0: {"input": 0.042, "cached_input": None, "output": 0.00},
    CM.jev_1_13_0: {"input": 0.042, "cached_input": None, "output": 0.00},
    CM.jev: {"input": 0.042, "cached_input": None, "output": 0.00},
}


def _scale_up(value):
    """Scale the value to avoid floating point errors."""
    return value * __PER_MILLION

def _scale_down(value):
    """Scale the value back down after avoiding floating point errors."""
    return value / __PER_MILLION

def _calc_price(tokens, rate) -> float:
    if tokens is None or rate is None:
        return 0
    if tokens == 0 or rate == 0:
        return 0
    return tokens * rate / __PER_MILLION


class Price(BaseModel):
    """USD price per million tokens.

    Attributes:
        name: The model name.
        input: Price of prompt tokens.
        cached_input: Price of cached prompt tokens, if available.
        output: Price of completion tokens.
        discount: The discount factor applied to the total cost.
    """
    input: NonNegativeFloat
    output: NonNegativeFloat
    name: str = Field(..., min_length=1)
    cached_input: NonNegativeFloat | None
    discount: PositiveFloat = 1

    def convert(
        self, 
        *,
        currency: str, 
        fx_rates: dict[str, float] | dict[Currency, float] = FX_RATES
    ) -> "Price":
        """Convert the price to the specified currency."""
        curr = Currency(currency)
        rate = fx_rates[curr]
        if rate == 0:
            raise ZeroDivisionError(f"FX rate for {curr} cannot be zero.")

        return Price(
            name=self.name,
            input=self.input / rate,
            cached_input=None if self.cached_input is None else self.cached_input / rate,
            output=self.output / rate,
            discount=self.discount,
        )


PRICES: dict[str, Price] = {
    model.lower(): Price(name=model, **price) 
    for model, price in _PRICES_USD.items()
}

def get_price(model: str, prices: dict[str, Price] = PRICES) -> Price:
    """Look up a model price without case sensitivity.

    Args:
        model: Model name.

    Returns:
        The model price.

    Raises:
        NotImplementedError: The model has no listed price.
    """
    try:
        return prices[model.strip().lower()]
    except KeyError:
        raise NotImplementedError(f"No token price is listed for model '{model}'.") from None


def calculate_token_cost(
    model: str, 
    input_tokens: int, 
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    currency: str = Currency.USD,
) -> float:
    """Calculate the cost of one model call.

    Args:
        model: Model name.
        input_tokens: Prompt token count.
        cached_input_tokens: The number of cached input tokens.
        output_tokens: Completion token count.
        currency: The currency code for the cost calculation.

    Returns:
        The total cost in the specified currency.

    Raises:
        NotImplementedError: The model has no listed price.
    """
    curr = Currency(currency)
    price = get_price(model)
    total = (
        _calc_price(input_tokens, price.input) +
        _calc_price(output_tokens, price.output) +
        _calc_price(cached_input_tokens, price.cached_input)
    )
    total /= price.discount
    return convert(total, currency=curr)


def convert(
    amount_usd: float, 
    *,
    currency: str, 
    rates: dict[str, float] | dict[Currency, float] = FX_RATES
) -> float:
    """Convert a US dollar amount into another currency.

    Args:
        amount_usd: Amount in US dollars.
        currency: Target currency code.
        rates: Exchange rates mapping currencies to their USD conversion rate.
            Defaults to hard-coded FX rates (USD, GDP, EUR).

    Returns:
        The amount in the target currency.

    Raises:
        KeyError: The currency is not in rates.
        ZeroDivisionError: The exchange rate for the currency is zero.
    """
    currency = Currency(currency)
    if currency not in rates:
        raise KeyError(f"Exchange rate for currency '{currency}' is not available.")
    rate = rates[currency]
    if rate == 0:
        raise ZeroDivisionError(f"Exchange rate for currency '{currency}' is zero.")
    return amount_usd / rate
