from .classes import Function, PromptTest
import json
from pathlib import Path
from pydantic import TypeAdapter


def read_json_func(json_path: str | Path) -> list['Function']:
    with open(json_path, 'r', encoding='utf-8') as arch:
        lists = json.load(arch)

    func = TypeAdapter(list[Function])
    val_func = func.validate_python(lists)

    return val_func


def read_json_prompt(json_path: str | Path) -> list['PromptTest']:
    with open(json_path, 'r', encoding='utf-8') as arch:
        lists = json.load(arch)

    prompt = TypeAdapter(list[PromptTest])
    val_prompt = prompt.validate_python(lists)

    return val_prompt


def promptformer(model, prompt):
    resul = []

    if isinstance(prompt, list):
        for p in prompt:
            txt_prompt = getattr(p, 'prompt', str(p))

            input_tensor = model.encode(txt_prompt)
            input_ids_list = input_tensor[0].tolist()
            logits = model.get_logits_from_input_ids(input_ids_list)

            resul.append(logits)

        return resul

    return None


def funcformer(model, func_name, prompt):
    if not isinstance(prompt, list):
        return None

    txt_contex = ""

    for f in func_name:
        name = getattr(f, 'name', '')
        desc = getattr(f, 'description', '')
        para = getattr(f, 'parameters', '')

        txt_contex += (
            f"- {name}: {desc}. Parameters: {para}\n"
        )

    function_names = [f.name for f in func_name]

    function_tokens = []

    for name in function_names:
        token_tensor = model.encode(name)
        token_ids = token_tensor[0].tolist()
        function_tokens.append(token_ids)

    print("testtee:", function_names)
    print("testee:", function_tokens)
    resul = []

    for p in prompt:
        txt_prompt = getattr(p, 'prompt', str(p))

        context = (
            f"{txt_contex}\n"
            f"Correct function: {txt_prompt}\n"
            f"Function:"
        )

        input_tensor = model.encode(context)
        input_ids_list = input_tensor[0].tolist()

        logits = model.get_logits_from_input_ids(input_ids_list)

        resul.append(logits)

    return resul


def predict_next_token_from_logits(model, logits: list[float]) -> dict:
    big_id = max(range(len(logits)), key=lambda i: logits[i])
    clean_txt = model.decode([big_id])

    return {
        "id": big_id,
        "clean_txt": clean_txt,
        "logit_valor": logits[big_id]
    }


def select_function(
    model,
    context_ids: list[int],
    function_token_sequences: list[list[int]],
) -> int:
    """Escole qual função bate com o contexto.., deiando limitade
    candidatas token por token :) constrained decoding).

    Args:
        model: a instância de Small_LLM_Model #LEMBRA DE TRADUZIR
        context_ids: tokens do contexto (ex: input_ids_list que
            funcformer já calcula pra cada prompt)
        function_token_sequences: pra cada função, a lista de
            token ids do nome dela (é o function_tokens que
            funcformer já monta)

    Returns:
        O índice, dentro de function_token_sequences, da função
        escolhida T.T
    """
    candidates = list(range(len(function_token_sequences)))
    generated: list[int] = []
    position = 0

    while len(candidates) > 1:
        expected_tokens = {
            function_token_sequences[i][position]
            for i in candidates
            if position < len(function_token_sequences[i])
        }

        if not expected_tokens:
            break

        logits = model.get_logits_from_input_ids(context_ids + generated)
        best_token = max(expected_tokens, key=lambda t: logits[t])
        generated.append(best_token)

        candidates = [
            i
            for i in candidates
            if position < len(function_token_sequences[i])
            and function_token_sequences[i][position] == best_token
        ]

        position += 1

    return candidates[0]