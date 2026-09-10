from pathlib import Path
from src.archives import load_func

path_json = Path(__file__).resolve().parent / 'call_me_maybe/datas/functions_definition.json'
minhas_funcoes = load_func(path_json)
for func in minhas_funcoes:
    print(f"Função encontrada: {func.name}")