"""Pydantic models used across the project."""
from typing import Any

from pydantic import BaseModel


class Typeinfo(BaseModel):
    """Type label of a parameter or return value (e.g. 'number')."""

    type: str


class Function(BaseModel):
    """One entry of functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, Typeinfo]
    returns: Typeinfo


class PromptTest(BaseModel):
    """One entry of function_calling_tests.json."""

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry of the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]


class TokenSets(BaseModel):
    """Token ids grouped by what they are allowed to be.

    Built once from the vocabulary, then reused for every prompt.
    """

    number_ids: list[int]
    string_ids: list[int]
    quote_ids: list[int]
    stop_ids: list[int]