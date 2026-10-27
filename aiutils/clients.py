"""Client instantiation utilities for OpenAI and other LLM clients."""
from enum import StrEnum
import os
from typing import TypeAlias
from unittest import case

from attr import dataclass


from openai import OpenAI, AsyncOpenAI


from .exceptions import ClientInstantiationError, InitializationError
from .models import Vendor


SupportedClients: TypeAlias = OpenAI | AsyncOpenAI


class EnvNames(StrEnum):
    GEMINI_API_KEY = "GEMINI_API_KEY"
    BRAVE_API_KEY = "BRAVE_API_KEY"
    OPENAI_API_KEY = "OPENAI_API_KEY"


def make_client_type(
    client_type: SupportedClients, 
    *, 
    api_key: str,
    base_url: str | None = None
) -> SupportedClients:
    if not api_key:
        raise InitializationError(f"api_key must be provided.")

    try:
        return client_type(
            api_key=api_key,
            base_url=base_url
        )
    except Exception as e:
        raise InitializationError(f"Failed to instantiate LLM client: {e}") from e


def client_type_kwargs(vendor: Vendor) -> dict:
    match vendor:
        case Vendor.OPENAI:
            api_key = EnvNames.OPENAI_API_KEY
            base_url = None
        case Vendor.GOOGLE:
            api_key = EnvNames.GEMINI_API_KEY
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        case Vendor.BRAVE:
            api_key = EnvNames.BRAVE_API_KEY
            base_url = "https://api.search.brave.com/res/v1"
        case _:
            raise ValueError(f"Unsupported vendor: {vendor}")
    api_key = os.getenv(api_key)
    if not api_key:
        raise EnvironmentError(f"API key for {vendor} not found in environment.")
    return {
        "api_key": api_key,
        "base_url": base_url
    }


def make_google_client() -> SupportedClients | None:
    return make_client_type(OpenAI, **client_type_kwargs(Vendor.GOOGLE))


def make_brave_client() -> SupportedClients | None:
    return make_client_type(OpenAI, **client_type_kwargs(Vendor.BRAVE))


def make_openai_client() -> SupportedClients | None:
    return make_client_type(OpenAI, **client_type_kwargs(Vendor.OPENAI))


def make_async_openai_client() -> SupportedClients | None:
    return make_client_type(AsyncOpenAI, **client_type_kwargs(Vendor.OPENAI))


@dataclass
class Client:
    async_client: SupportedClients
    client: SupportedClients
    vendor: Vendor = Vendor.OPENAI

    def set_vendor(self, vendor: str):
        self.vendor = Vendor(vendor.lower())
        match self.vendor:
            case Vendor.OPENAI:
                self.client = make_openai_client()
                self.async_client = make_async_openai_client()
            case Vendor.GOOGLE:
                self.client = make_google_client()
                # TODO
                self.async_client = make_async_openai_client()
            case Vendor.BRAVE:
                self.client = make_brave_client()
                # TODO
                self.async_client = make_async_openai_client()
            case _:
                raise ValueError(f"Unsupported vendor: {vendor}")
        if self.client is None:
            raise RuntimeError(f"Sync client not available for {vendor}.")
        if self.async_client is None:
            raise RuntimeError(f"Async client not available for {vendor}.")


def make_client(
    vendor: str = Vendor.OPENAI,
    client: SupportedClients = make_openai_client(),
    async_client: SupportedClients = make_async_openai_client(),
):
    vendor = Vendor(vendor.lower())

    try:
        return Client(
            client=client,
            async_client=async_client,
            vendor=vendor
        )
    except RuntimeError as e:
        raise InitializationError(f"Failed to make client for vendor {vendor}: {e}") from e
