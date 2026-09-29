from .functions import Function, PromptTest
import json
from pathlib import Path
from pydantic import TypeAdapter
import torch


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


def predict_next_token_from_logits(self, logits: list[float]) -> dict:
    logits_tensor = torch.tensor(logits)

    big_id = torch.argmax(logits_tensor).item()
    puretoken = self._tokenizer._convert_id_to_token(big_id)
    clean_txt = self.decode([big_id])

    return {
        "id": big_id,
        "puretoken": puretoken,
        "clean_txt": clean_txt,
        "logit_valor": float(logits_tensor[big_id].item())
    }

