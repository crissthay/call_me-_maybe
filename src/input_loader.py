import json
from pathlib import Path
from pydantic import TypeAdapter
from .classes import Function, PromptTest

def read_json_func(json_path: str | Path) -> list[Function]:
    """Read functions_definition.json and validate it.

    Args:
        json_path: path to the functions definition file.

    Returns:
        The list of validated Function objects.
    """
    with open(json_path, 'r', encoding='utf-8') as arch:
        lists = json.load(arch)
    return TypeAdapter(list[Function]).validate_python(lists)

def read_json_prompt(json_path: str | Path) -> list[PromptTest]:
    """Read function_calling_tests.json and validate it.

    Args:
        json_path: path to the prompts file.

    Returns:
        The list of validated PromptTest objects.
    """
    with open(json_path, 'r', encoding='utf-8') as arch:
        lists = json.load(arch)
    return TypeAdapter(list[PromptTest]).validate_python(lists)
