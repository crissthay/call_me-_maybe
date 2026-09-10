from pydantic import BaseModel, TypeAdapter, ValidationError
import json
from llm_sdk.model import Small_LLM_Model

class Typeinfo (BaseModel):
    type: str

class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, Typeinfo]
    returns: Typeinfo

class PromptTest(BaseModel):
    prompt: str

caminho2 = '/Users/liza/Documents/for transfer - documents/Promised Files/tester/call_me_maybe/datas/function_calling_tests.json'
with open(caminho2, "r", encoding="utf-8") as f:
    tests = json.load(f)
for test in tests:
    print(test["prompt"]) #para apagar

caminho = '/Users/liza/Documents/for transfer - documents/Promised Files/tester/call_me_maybe/datas/functions_definition.json'
with open(caminho, 'r', encoding='utf-8') as arquivo:
    lista_de_dicionarios = json.load(arquivo)

lista_validada = TypeAdapter(list[Function]).validate_python(lista_de_dicionarios)

#tester
print(type(lista_validada))
print(type(lista_validada[0]))
print(lista_validada[0].name)
print(lista_validada[0].parameters['a'].type)