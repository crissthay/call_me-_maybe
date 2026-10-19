import json
from typing import Any
import numpy as np
from .classes import Function, FunctionCallResult, TokenSets
 
NUMBER_CHARS = set("0123456789.-")
MAX_NUMBER_TOKENS = 12
MAX_STRING_TOKENS = 40

def load_vocab(path: str) -> dict[int, str]:
    """Load the vocabularyu file as {token_id: readable_text}.

    The file maps token text -> id. Byte-level tokenizers write a space
    as 'Ġ' and a newline as 'Ċ', so we translate them back.

    Args:p
        path: result of model.get_path_to_vocab_file().

    Returns:
        Dict from token id to the text that token represents.
    """
    with open(path, 'r', encoding='utf-8') as file:
        raw: dict[str, int] = json.load(file)
    return {
        tid: text.replace('Ġ', ' ').replace('Ċ', '\n')
        for text, tid in raw.items()
    }

def build_token_sets(vocab: dict[int, str]) -> TokenSets:
    """Classify every token once, so we do not redo it per prompt.
 
    Args:
        vocab: result of load_vocab.

    Returns:
        TokenSets with the ids allowed for numbers, strings, the closing
        quote, and the tokens that signal "the number ended".
    """
    number_ids = [
        i for i, t in vocab.items()
        if t and all(c in NUMBER_CHARS for c in t)
    ]
    string_ids = [
        i for i, t in vocab.items()
        if t and t.isprintable() and '"' not in t
        and '\\' not in t and not t.startswith('<|')
    ]
    quote_ids = [i for i, t in vocab.items() if t == '"']
    stop_ids = [i for i, t in vocab.items() if t in (',', '}')]
    return TokenSets(
        number_ids=number_ids,
        string_ids=string_ids,
        quote_ids=quote_ids,
        stop_ids=stop_ids,
    )
 

def encode_ids(model: Any, text: str) -> list[int]:
    """Turn text into a flat list of token ids using the SDK."""
    return list(model.encode(text)[0].tolist())

def best_allowed(logits: list[float], allowed: list[int]) -> int:
    """Return the allowed token id with the highest logit.

    Args:
        logits: one score per token in the vocabulary.
        allowed: the only ids we accept right now.

    Returns:
        The best id among "allowed".
    """
    scores = np.asarray(logits)
    allowed_arr = np.asarray(allowed)
    return int(allowed_arr[np.argmax(scores[allowed_arr])])

def select_sequence(
    model: Any,
    context_ids: list[int],
    sequences: list[list[int]],
) -> int:
    """Pick one of a fixed list of candidates, token by token.

    Used for function names and for true/false. Each step keeps only
    the candidates whose next token matches the token the model chose.

    Args:
        model: the Small_LLM_Model instance.
        context_ids: tokens of the prompt so far.
        sequences: token ids of each candidate.

    Returns:
        Index (in "sequences") of the chosen candidate.
    """
    candidates = list(range(len(sequences)))
    generated: list[int] = []
    position = 0

    while len(candidates) > 1:
        expected = list({
            sequences[i][position]
            for i in candidates if position < len(sequences[i])
        })
        if not expected:
            break
        logits = model.get_logits_from_input_ids(context_ids + generated)
        chosen = best_allowed(logits, expected)
        generated.append(chosen)
        candidates = [
            i for i in candidates
            if position < len(sequences[i]) and sequences[i][position] == chosen
        ]
        finished = [i for i in candidates if len(sequences[i]) == position + 1]
        if finished:
            return finished[0]
        position += 1
 
    return candidates[0]

def _can_extend(current: str, piece: str) -> bool:
    """Check that adding "piece" keeps "current" a valid number prefix."""
    for char in piece:
        if char == '-' and current != '':
            return False
        if char == '.' and ('.' in current or current in ('', '-')):
            return False
        current += char
    return True

def generate_number(
    model: Any,
    vocab: dict[int, str],
    sets: TokenSets,
    context_ids: list[int],
) -> tuple[float, list[int]]:
    """Generate a JSON number token by token.

    Only digit/dot/minus tokens are allowed (following number rules).
    Once the number has at least one digit, the 'stop' tokens (',' or
    '}') also become allowed: if the model picks one, the number ended.
 
    Returns:
        The number and the token ids that were generated.
    """
    generated: list[int] = []
    text = ''

    for _ in range(MAX_NUMBER_TOKENS):
        logits = model.get_logits_from_input_ids(context_ids + generated)
        allowed = [
            i for i in sets.number_ids if _can_extend(text, vocab[i])
        ]
        if text and text[-1].isdigit():
            allowed = allowed + sets.stop_ids
        chosen = best_allowed(logits, allowed)
        if chosen in sets.stop_ids:
            break
        generated.append(chosen)
        text += vocab[chosen]

    try:
        return float(text), generated
    except ValueError:
        return 0.0, generated

def generate_string(
    model: Any,
    sets: TokenSets,
    context_ids: list[int],
) -> tuple[str, list[int]]:
    """Generate the content of a JSON string (without the quotes).

    Any token without a quote/backslash is allowed. When the model
    picks the closing quote, the string ends.

    Returns:
        The decoded text and the token ids that were generated.
    """
    generated: list[int] = []

    for _ in range(MAX_STRING_TOKENS):
        logits = model.get_logits_from_input_ids(context_ids + generated)
        chosen = best_allowed(logits, sets.string_ids + sets.quote_ids)
        if chosen in sets.quote_ids:
            break
        generated.append(chosen)

    return str(model.decode(generated)), generated

def build_context(functions: list[Function], prompt: str) -> str:
    """Build the text that explains the task to the model."""
    lines = ['Available functions:']
    for func in functions:
        params = ', '.join(
            f'{n}: {t.type}' for n, t in func.parameters.items()
        )
        lines.append(f'- {func.name}({params}): {func.description}')
    lines.append(f'\nUser request: {prompt}')
    return '\n'.join(lines)

def resolve_prompt(
    model: Any,
    vocab: dict[int, str],
    sets: TokenSets,
    functions: list[Function],
    prompt: str,
) -> FunctionCallResult:
    """Turn one prompt into a function name plus its arguments.
 
    Step 1: choose the function (select_sequence over the names).
    Step 2: for each parameter, write the fixed JSON parts ourselves
    (key, colon, quotes) and let the model generate ONLY the value,
    constrained by the parameter type.
 
    Returns:
        The validated result for this prompt.
    """
    context = build_context(functions, prompt)
    base = encode_ids(model, context + '\nThe best function is:')
 
    names = [encode_ids(model, ' ' + f.name) for f in functions]
    function = functions[select_sequence(model, base, names)]

    ids = base + names[functions.index(function)]
    ids += encode_ids(model, '\nArguments as JSON: {')
    values: dict[str, Any] = {}
    last = len(function.parameters) - 1

    for pos, (pname, ptype) in enumerate(function.parameters.items()):
        kind = ptype.type
        quote = '"' if kind == 'string' else ''
        ids += encode_ids(model, f'"{pname}": {quote}')
 
        if kind in ('number', 'integer'):
            number, new = generate_number(model, vocab, sets, ids)
            values[pname] = int(number) if kind == 'integer' else number
        elif kind == 'string':
            values[pname], new = generate_string(model, sets, ids)
        elif kind == 'boolean':
            options = [
                encode_ids(model, 'true'), encode_ids(model, 'false')
            ]
            index = select_sequence(model, ids, options)
            values[pname], new = (index == 0), options[index]
        else:
            raise ValueError(f'Unsupported parameter type: {kind}')
 
        ids += new
        ids += encode_ids(model, f'{quote}' + (', ' if pos < last else '}'))

    return FunctionCallResult(
        prompt=prompt, name=function.name, parameters=values
    )
