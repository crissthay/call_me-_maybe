from .functions import Function, PromptTest
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

"""
ver dps
def promptformer(model, prompt):
    if hasattr(prompt, 'encode'):
        return model.encode(prompt)
    return None"""

def promptformer(model, prompt):
    if isinstance(prompt, list):
        for p in prompt:
            resul = []
            txt_prompt = getattr(p, 'prompt', str(p)) 
            resul.append(model.enconde(txt_prompt))
        return resul
            # o getattr e para procurar dentro do texto
    
    if hasattr(prompt, 'encode'):
        return model.encode(prompt)
    return None

"""acho que essa ta certa mas n tenho um pc bom :()
def promptformer(model, prompt):
    if isinstance(prompt, list):
        for p in prompt:
            resul = []
            txt_prompt = getattr(p, 'prompt', str(p)) 
            input_tensor = model.encode(txt_prompt)
            input_ids_list = input_tensor[0].tolist()
            logits = model.get_logits_from_input_ids(input_ids_list)
            
            resul.append(logits)
        return resul
        
    return None"""
