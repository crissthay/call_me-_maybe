"""Constrained function-call decoding and argument extraction."""

import json
import re
from typing import Any

import numpy as np

from src.classes import Function, FunctionCallResult, TokenSets


MAX_NUMBER_TOKENS = 16


def load_vocab(path: str) -> dict[str, int]:
    """Load the model vocabulary from its JSON file."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def build_token_sets(vocab: dict[str, int]) -> TokenSets:
    """Build reusable groups of token IDs."""
    number_ids: list[int] = []
    string_ids: list[int] = []
    quote_ids: list[int] = []
    stop_ids: list[int] = []

    for token, token_id in vocab.items():
        clean = token.replace("Ġ", " ").replace("Ċ", "\n").replace("ċ", "\n")
        if clean == '"':
            quote_ids.append(token_id)
        elif clean in (",", "}"):
            stop_ids.append(token_id)
        elif clean and all(char in "0123456789.-" for char in clean):
            number_ids.append(token_id)
        elif clean.isprintable() and '"' not in clean and not token.startswith("<|"):
            string_ids.append(token_id)

    return TokenSets(
        number_ids=number_ids,
        string_ids=string_ids,
        quote_ids=quote_ids,
        stop_ids=stop_ids,
    )


def encode_ids(model: Any, text: str) -> list[int]:
    """Encode text with the tokenizer exposed by the model."""
    return model.encode(text)[0].tolist()


def best_allowed(logits: list[float], allowed_ids: list[int]) -> int:
    """Return the highest-logit token among permitted token IDs."""
    if not allowed_ids:
        raise ValueError("There are no permitted tokens.")
    scores = np.asarray(logits)
    ids = np.asarray(allowed_ids, dtype=int)
    return int(ids[np.argmax(scores[ids])])


def select_sequence(
    model: Any,
    context: list[int],
    candidates: list[list[int]],
) -> list[int]:
    """Select one complete candidate sequence using constrained decoding."""
    remaining = [sequence for sequence in candidates if sequence]
    generated: list[int] = []

    while remaining:
        next_ids = list({
            sequence[len(generated)]
            for sequence in remaining
            if len(sequence) > len(generated)
        })
        if not next_ids:
            break

        logits = model.get_logits_from_input_ids(context + generated)
        token = best_allowed(logits, next_ids)
        generated.append(token)

        remaining = [
            sequence for sequence in remaining
            if sequence[:len(generated)] == generated
        ]
        if generated in remaining:
            return generated

    raise ValueError("The model could not select a permitted candidate.")


def generate_number(
    model: Any,
    context: list[int],
    token_sets: TokenSets,
    integer: bool = False,
) -> tuple[int | float, list[int]]:
    """Generate a numeric argument with restricted vocabulary tokens."""
    generated: list[int] = []

    for _ in range(MAX_NUMBER_TOKENS):
        logits = model.get_logits_from_input_ids(context + generated)
        allowed = token_sets.number_ids.copy()
        if generated:
            allowed.extend(token_sets.stop_ids)

        token = best_allowed(logits, allowed)
        if token in token_sets.stop_ids:
            break
        generated.append(token)

    text = model.decode(generated).strip()
    if not text or text in ("-", "."):
        raise ValueError("The model generated an invalid number.")

    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value: {text!r}") from exc

    if not np.isfinite(value):
        raise ValueError("The generated number is not finite.")
    if integer and not value.is_integer():
        raise ValueError("An integer argument was expected.")
    if value.is_integer():
        return int(value), generated
    return value, generated


def _quoted_values(prompt: str) -> list[str]:
    """Return text enclosed in matching single or double quotes."""
    matches = re.findall(r'"([^"]*)"|\'([^\']*)\'', prompt)
    values = [double if double else single for double, single in matches]
    return list(dict.fromkeys(values))


def _word_values(prompt: str) -> list[str]:
    """Return distinct word-like values from the request."""
    return list(dict.fromkeys(re.findall(r"[A-Za-z0-9_]+", prompt)))


def _string_candidates(prompt: str, parameter: str) -> list[str]:
    """Build literal candidates; constrained decoding chooses between them."""
    quoted = _quoted_values(prompt)
    words = _word_values(prompt)

    if parameter == "source_string":
        candidates = quoted or words
    elif parameter == "regex":
        candidates = [r"\d+", "[aeiouAEIOU]", r"\w+", ".*"]
        for word in words:
            if len(word) > 1:
                candidates.extend((re.escape(word), r"\b" + re.escape(word) + r"\b"))
    elif parameter == "replacement":
        candidates = quoted + words
        if re.search(r"\basterisks?\b", prompt, flags=re.IGNORECASE):
            candidates.append("*")
    else:
        candidates = quoted + words

    return list(dict.fromkeys(value for value in candidates if value != ""))


def generate_string(
    model: Any,
    context: list[int],
    prompt: str,
    parameter: str,
    token_sets: TokenSets,
) -> tuple[str, list[int]]:
    """Select a string candidate instead of free-running generation."""
    if not token_sets.quote_ids:
        raise ValueError('The vocabulary has no standalone double-quote token.')

    values = _string_candidates(prompt, parameter)
    if not values:
        raise ValueError(f"No candidate text found for parameter {parameter!r}.")

    quote_ids = encode_ids(model, '"')
    if not quote_ids:
        raise ValueError("The tokenizer could not encode a closing quote.")

    sequences = [encode_ids(model, value) + quote_ids for value in values]
    selected = select_sequence(model, context, sequences)

    for value, sequence in zip(values, sequences):
        if selected == sequence:
            return value, selected
    raise ValueError("Selected string did not match a candidate.")


def generate_boolean(model: Any, context: list[int]) -> tuple[bool, list[int]]:
    """Select either true or false using constrained decoding."""
    options = [encode_ids(model, "true"), encode_ids(model, "false")]
    selected = select_sequence(model, context, options)
    decoded = model.decode(selected).strip()
    if decoded not in ("true", "false"):
        raise ValueError("The model generated an invalid boolean.")
    return decoded == "true", selected


def build_context(functions: list[Function], prompt: str) -> str:
    """Describe available functions and append the user's request."""
    descriptions: list[str] = []
    for function in functions:
        parameters = ", ".join(
            f"{name}: {details.type}"
            for name, details in function.parameters.items()
        )
        descriptions.append(
            f"{function.name}({parameters}): {function.description}"
        )

    return (
        "Choose the best function for the user's request.\n"
        "Extract argument values only. Do not execute the function.\n"
        + "\n".join(descriptions)
        + f"\nUser request: {prompt}\n"
        + "Select the function name:\n"
    )


def resolve_prompt(
    model: Any,
    vocab: dict[str, int],
    token_sets: TokenSets,
    functions: list[Function],
    prompt: str,
) -> FunctionCallResult:
    """Choose a function and construct its typed arguments."""
    del vocab  # The vocabulary is already represented by token_sets.

    context = encode_ids(model, build_context(functions, prompt))
    function_sequences = [encode_ids(model, item.name) for item in functions]
    selected_function = select_sequence(model, context, function_sequences)
    function_name = model.decode(selected_function).strip()

    function = next(
        (item for item in functions if item.name == function_name),
        None,
    )
    if function is None:
        raise ValueError(f"Unknown function selected: {function_name!r}")

    context += selected_function
    context += encode_ids(model, '\n{"')
    arguments: dict[str, Any] = {}

    for index, (name, details) in enumerate(function.parameters.items()):
        if index:
            context += encode_ids(model, ',"')
        context += encode_ids(model, name + '":')

        if details.type == "string":
            context += encode_ids(model, '"')
            value, generated = generate_string(
                model, context, prompt, name, token_sets
            )
        elif details.type in ("number", "integer"):
            value, generated = generate_number(
                model,
                context,
                token_sets,
                integer=details.type == "integer",
            )
        elif details.type == "boolean":
            value, generated = generate_boolean(model, context)
        else:
            raise ValueError(
                f"Unsupported parameter type {details.type!r} for {name!r}."
            )

        arguments[name] = value
        context += generated

    return FunctionCallResult(
        prompt=prompt,
        name=function.name,
        parameters=arguments,
    )