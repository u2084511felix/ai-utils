import re
import ast
from typing import Any, Set, Dict, Tuple, List, Optional, Union, Type
from pydantic import BaseModel, create_model, Field
from aiutils.openai_config import Generate, TextModels, cclient, create_generator_module
import json
import os
import inspect
import sys
import subprocess
from datamodel_code_generator import InputFileType, DataModelType, generate
from aiutils import client
import pdb

def run_script_action(action):
    os.system(action)
    # subprocess.run(action, shell=True, check=True,
    #                text=True, capture_output=True)


def get_caller_script_dir():
    stack = inspect.stack()

    for frame in reversed(stack):
        caller_file = frame.filename
        if "modules.py" not in caller_file:
            return os.path.dirname(os.path.abspath(caller_file))

    return os.getcwd()


def save_json_file(data, file_name):
    """Saves JSON in the same directory as the script that originally called modules.py."""
    caller_dir = get_caller_script_dir() + "/tmp/"
    os.makedirs(caller_dir, exist_ok=True)

    file_path = os.path.join(caller_dir, file_name)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"File saved to: {file_path}")


async def legacy_structured_output(prompt, schema, system_msg="default") -> str:
    """Returns a JSON string"""

    if (system_msg == "default"):
        system_message = "Supply the function variables for the given function according to the instruction."

    else:
        system_message = system_msg

    LegacyStructuredOutput = Generate()
    new_schema = json.loads(schema)

    function_name = new_schema.get("name")
    LegacyStructuredOutput.tool_choice = {
        "type": "function", "function": {"name": function_name}}

    LegacyStructuredOutput.tools = [
        {
            "type": "function",
            "function": new_schema
        }
    ]

    legacy_structured_output = await LegacyStructuredOutput.function_call_legacy_structured_output(system_message, prompt)

    legacy_structured_output = legacy_structured_output[0].function.arguments
    output_json = json.loads(legacy_structured_output)
    
    #save_json_file(output_json, "finaloutput.json")
    
    return legacy_structured_output


async def generate_legacy_structured_output_schema(json_object, system_msg="default") -> str:
    """Returns a JSON string"""

    transforn_prompt = "Transform this JSON object: " + str(json_object)
    SchemaGenerator = Generate()

    if (system_msg == "default"):
        sys_msg = f"""Generate a JSON schema for the given instruction converting from an existing data model outline, using the function."""

    else:
        sys_msg = system_msg

    SchemaGenerator.tools = [
        {
            "type": "function",
            "function": {
                "name": "Tool_Schema",
                "description": "Schema for generating tool schemas. Do not use $ref in the schema. Use the 'type' and 'properties' fields to define the schema.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"
                        },
                        "description": {
                            "type": "string"
                        },
                        "parameters": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string"
                                        },
                                        "properties": {
                                            "type": "object",
                                            "enum": [],
                                            "additionalProperties": False
                                        },
                                        "required": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        }
                                    },
                                    "required": ["type", "properties", "required"],
                                    "additionalProperties": False
                                }
                            ]
                        }
                    },
                    "required": ["name", "description", "parameters"]
                }
            }
        }
    ]

    SchemaGenerator.tool_choice = {
        "type": "function", "function": {"name": "Tool_Schema"}}

    generated_schema = await SchemaGenerator.function_call_legacy_structured_output(sys_msg, transforn_prompt)

    generated_schema = generated_schema[0].function.arguments
    output_json = json.loads(generated_schema)

    # if debug:
    #     #save_json_file(output_json, "schema_output.json")
    #     pass

    return generated_schema


async def generate_structured_output_schema(json_object, system_msg="default"):

    transforn_prompt = "Transform this JSON object: " + str(json_object)
    SchemaGenerator = Generate()

    if (system_msg == "default"):
        sys_msg = f"""Generate a JSON schema for the given instruction converting from an existing data model outline, using the function."""

    else:
        sys_msg = system_msg

    SchemaGenerator.tools = [
        {
            "type": "function",
            "function": {
                "name": "Tool_Schema",
                "description": "Schema for generating schemas from a draft json model. The first 'name' parameter, should relate to the content model, and the 'description' parameter should describe the purpose of the content model. The 'schema' parameter should produce named objects that define a valid schema for the content model. Each parameter in the content model should be under a named object in the 'schema' parameter. All name properties must be single word which matches the pattern: '^[a-zA-Z0-9_-]+$'. The 'schema' object should output schema objects with a maximum nesting level of 5. Any objects inside of named schema objects should contain a required array listing all required elements in the object and 'additionalProperties' = 'false'. Likewise any arrays in named schema objects",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"
                        },
                        "description": {
                            "type": "string"
                        },
                        "schema": {
                            "type": "object",
                            "description": "schema object containing named schema objects that define each property the content model schema. Maximum nesting of 5 for each named schema object",
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string"
                                        },
                                        "properties": {
                                            "oneOf": [
                                                {
                                                    "name": {
                                                        "decription": "The named schema object for one element of the content model. Maximum nesting of 5.",
                                                        "type": "object",
                                                        "oneOf": [
                                                            {
                                                                "type": "object",
                                                                "properties": {
                                                                    "description": {
                                                                        "type": "string",
                                                                        "description": "the description of the named schema object"
                                                                    },
                                                                    "type": {
                                                                        "description": "the type of an element in a named schmea object",
                                                                        "type": "string",
                                                                        "enum": ["string", "number", "boolean", "integer", "object", "array"]
                                                                    },
                                                                    "properties": {
                                                                        "type": "object",
                                                                        "enum": [],
                                                                        "additionalProperties": False
                                                                    },
                                                                    "required": {
                                                                        "type": "array",
                                                                        "items": {
                                                                            "type": "string"
                                                                        }
                                                                    }
                                                                },
                                                                "required": ["description", "type"],
                                                                "additionalProperties": False
                                                            }
                                                        ],
                                                    }
                                                }
                                            ]
                                        },
                                        "required": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        }
                                    },
                                    "required": ["type", "properties", "required"],
                                    "additionalProperties": False
                                }

                            ]
                        }

                    },
                    "required": ["name", "description", "schema"]
                }
            }
        }
    ]

    SchemaGenerator.tool_choice = {
        "type": "function", "function": {"name": "Tool_Schema"}}

    generated_schema = await SchemaGenerator.function_call_legacy_structured_output(sys_msg, transforn_prompt)

    generated_schema = generated_schema[0].function.arguments

    output_json = json.loads(generated_schema)
    #save_json_file(output_json, "schema_output.json")

    return generated_schema


async def structured_outputs_generator(transforn_prompt, schema, system_msg="default", input_type='json', module_name='', params={}, model='', vendor='openai'):

    cclient.set_vendor(vendor)

    if vendor == 'google':
        max_tokens = 65536
    else:
        if model in ["gpt-5.1", "gpt-5.2"]:
            max_tokens = 128000
        else:
            max_tokens = 32768

    if (system_msg == "default"):
        sys_msg = f"""Generate a JSON schema for the given content model."""
    else:
        sys_msg = system_msg

    if (params is {}):
        module = create_generator_module(
            max_tokens=max_tokens, system_message=sys_msg)

    else:
        module = create_generator_module(**params)

    try:
        if model != '':
            structured_output = await module.structured_output(system_msg, transforn_prompt, schema=schema, input_type=input_type, module_name=module_name, model=model)
        else:
            structured_output = await module.structured_output(system_msg, transforn_prompt, schema=schema, input_type=input_type, module_name=module_name)

    except Exception as e:
        return e

    output_json = json.loads(structured_output)
    #save_json_file(output_json, "latest_structured_outout.json")
    return output_json


def pydantic_model_generator(json_input, output_file, template_dir):
    """
    Generates pydantic models class using datamodel-codegen and outputs to output_dir

    """
    action_string = f'datamodel-codegen --input {
        json_input} --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel --output {output_file} --custom-template-dir {template_dir}'

    try:
        print('running action script')
        run_script_action(action_string)
        print('success, rebuilding model now...')
        add_model_rebuild_calls(output_file)

    except Exception as e:
        print(e)
        return False

    return True


def reformat_generated_schema(schema_object, input_file, output_model_filepath):

    reformatted_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": schema_object["name"],
        "description": schema_object["description"],
        "type": "object",
        "properties": schema_object["schema"]["properties"],
        "required": schema_object["schema"]["required"],
        "additionalProperties": False
    }

    with open(input_file, "w") as f:
        json.dump(reformatted_schema, f, indent=4)

    template_file = os.getcwd() + '/app/models/custom_templates/'

    return (pydantic_model_generator(input_file, output_model_filepath, template_file))


def find_recursive_patterns(filepath: str) -> Dict[str, List[str]]:
    """
    Finds classes containing List[BaseModel] or Dict[..., BaseModel] fields.
    Direct BaseModel references are excluded from the results.
    """
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())

    # Identify all BaseModel-derived classes
    base_models = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(isinstance(b, ast.Name) and b.id == 'BaseModel' for b in node.bases):
                base_models.add(node.name)

    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in base_models:
            container_class = node.name
            collection_fields = []

            for body_item in node.body:
                if isinstance(body_item, ast.AnnAssign) and isinstance(body_item.target, ast.Name):
                    field_name = body_item.target.id
                    annotation = body_item.annotation

                    # Handle both new (3.9+) and old subscript syntax
                    if isinstance(annotation, ast.Subscript):
                        container_type, value_type = None, None

                        # Extract container type (List/Dict)
                        if isinstance(annotation.value, ast.Name):
                            container_type = annotation.value.id
                        elif isinstance(annotation.value, ast.Attribute):
                            container_type = annotation.value.attr  # For typing.List/typing.Dict

                        # Extract contained type
                        slice_node = annotation.slice
                        if isinstance(slice_node, ast.Index):  # Python < 3.9
                            slice_node = slice_node.value

                        if container_type == 'List':
                            if isinstance(slice_node, ast.Name):
                                value_type = slice_node.id
                        elif container_type == 'Dict':
                            if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) > 1:
                                # Dict value type
                                value_type_node = slice_node.elts[1]
                                if isinstance(value_type_node, ast.Name):
                                    value_type = value_type_node.id

                        # Check if contained type is a BaseModel
                        if value_type and value_type in base_models:
                            collection_fields.append(field_name)

            if collection_fields:
                result[container_class] = collection_fields

    return result


def add_model_rebuild_calls(filepath: str):
    """
    Adds ClassName.model_rebuild() calls immediately after class definitions
    that contain collections of other Pydantic models.
    Preserves original formatting and avoids duplicate calls.
    """
    target_classes = find_recursive_patterns(filepath).keys()

    with open(filepath, 'r') as f:
        content = f.read()

    tree = ast.parse(content)
    lines = content.split('\n')
    insertions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in target_classes:
            class_name = node.name
            end_line = node.end_lineno  # 1-based line number

            # Check if rebuild call already exists immediately after class
            if end_line < len(lines):
                next_line = lines[end_line].strip()
                if f"{class_name}.model_rebuild()" in next_line:
                    continue

            # Record insertion point (end_line is last line of class)
            insertions.append((end_line, f"{class_name}.model_rebuild()"))

    # Apply insertions in reverse order to preserve line numbers
    for line_num, call in sorted(insertions, reverse=True, key=lambda x: x[0]):
        # Convert 1-based line number to 0-based list index
        insert_index = line_num  # Lines after class end at line_num (1-based)
        lines.insert(insert_index, call)

    # Write back to file with original line endings
    with open(filepath, 'w', newline='') as f:
        f.write('\n'.join(lines))


async def GeneratePydanticModelStructuredOutput(prompt, schema_object_filepath, new_output_schema_filepath, new_pydantic_model_filepath):

    schema_object = {}
    with open(schema_object_filepath) as f:
        schema_object = json.load(f)

    input_file = new_output_schema_filepath
    schema_title = schema_object["name"]
    module_name = os.path.splitext(
        os.path.basename(new_pydantic_model_filepath))[0]

    reformat_generated_schema(
        schema_object, input_file, new_pydantic_model_filepath)

    final_output = await structured_outputs_generator(
        prompt, schema_title, input_type='pydantic', module_name=module_name)

    return final_output



async def responses_structured_output(model, system_message, prompt, pydantic_model):

    response = client.responses.parse(
        model=model,
        instructions=system_message,
        input=prompt,
        reasoning={
            "effort": "low"
        },
        text_format=pydantic_model,
    )

    #json = response.output_parsed.model_dump()
    return response.output_parsed 
