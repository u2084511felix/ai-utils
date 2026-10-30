"""Structured-output request validation."""

import importlib
import json
from copy import deepcopy

from jsonschema import Draft7Validator, Draft202012Validator
from pydantic import BaseModel


_DIALECTS = {
    "https://json-schema.org/draft/2020-12/schema": Draft202012Validator,
    "http://json-schema.org/draft-07/schema#": Draft7Validator,
    "http://json-schema.org/draft-07/schema": Draft7Validator,
}


def validate_schema(schema: dict | str) -> dict:
    """Validate a named JSON Schema wrapper.

    Args:
        schema: A dictionary or JSON string with name and schema fields.

    Returns:
        A validated wrapper independent of the supplied object.

    Raises:
        TypeError: The input type is unsupported.
        ValueError: The wrapper or dialect is invalid.
        jsonschema.SchemaError: The enclosed schema is invalid.
        json.JSONDecodeError: The JSON string is malformed.
    """
    if isinstance(schema, str):
        wrapper = json.loads(schema)
    elif isinstance(schema, dict):
        wrapper = deepcopy(schema)
    else:
        raise TypeError("schema must be a dictionary or JSON string.")
    if not isinstance(wrapper, dict):
        raise ValueError("A named schema wrapper must be an object.")
    json.dumps(wrapper, allow_nan=False)
    name = wrapper.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("The schema wrapper requires a nonempty name.")
    body = wrapper.get("schema")
    if not isinstance(body, dict):
        raise ValueError("The schema wrapper requires an object schema.")
    if set(wrapper) - {"name", "schema", "strict", "description"}:
        raise ValueError("The schema wrapper contains unsupported fields.")
    if "strict" in wrapper and type(wrapper["strict"]) is not bool:
        raise ValueError("The strict field must be a boolean.")
    if "description" in wrapper and not isinstance(wrapper["description"], str):
        raise ValueError("The description field must be a string.")
    dialect = body.get("$schema", "https://json-schema.org/draft/2020-12/schema")
    if not isinstance(dialect, str) or dialect not in _DIALECTS:
        raise ValueError("Only JSON Schema Draft 2020-12 and Draft 7 are supported.")
    _DIALECTS[dialect].check_schema(body)
    return wrapper


def validate_model(schema: type[BaseModel] | str, module_name: str = "") -> type[BaseModel]:
    """Validate a Pydantic class or a named class from a module.

    Args:
        schema: Model class or model name when module_name is supplied.
        module_name: Optional module that exports the named model.

    Returns:
        A supported Pydantic model class.

    Raises:
        TypeError: The input does not identify a supported model class.
        ValueError: The named model cannot be found.
    """
    if not isinstance(module_name, str):
        raise TypeError("module_name must be a string.")
    if module_name:
        if not isinstance(module_name, str) or not isinstance(schema, str) or not schema.strip():
            raise TypeError("Named Pydantic lookup requires a module name and a model name.")
        try:
            schema = getattr(importlib.import_module(module_name), schema)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Cannot find Pydantic model: {error}") from error
    if not isinstance(schema, type) or not issubclass(schema, BaseModel) or schema is BaseModel:
        raise TypeError("Pydantic input requires a Pydantic model class.")
    Draft202012Validator.check_schema(schema.model_json_schema())
    return schema
