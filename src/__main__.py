import os
from .input_loader import read_json_func, read_json_prompt, promptformer, funcformer
from llm_sdk.llm_sdk.__init__ import Small_LLM_Model
import torch


if __name__ == "__main__":
    pastaatual = os.path.dirname(os.path.abspath(__file__))
    path_json = os.path.abspath(
        os.path.join(pastaatual, "..", "datas", "functions_definition.json")
    )
    path_prompt = os.path.abspath(
        os.path.join(pastaatual, "..", "datas", "function_calling_tests.json")
    )
    func = read_json_func(path_json)
    pro = read_json_prompt(path_prompt)
    
    print("JSON:\n")
    for line in func:
        print("Name =", line.name)
        print("Description =", line.description)
        print("Parameters =", line.parameters)
        print("Returns =", line.returns)
       

    print("\nPROMPT:\n")
    for line in pro:
        print(line)
    

    print("\n === TEST encode ====")
    model = Small_LLM_Model()
    res = promptformer(model, pro)
    
    if res is not None:
        print("\n=== RESULTADOS DA PREVISÃO DO MODELO ===\n")
        for i, logits_da_frase in enumerate(res):
            logits_tensor = torch.tensor(logits_da_frase)
            next_token_id = torch.argmax(logits_tensor).item()
            palavra_prevista = model.decode([next_token_id])
            
            print(f"Prompt {i+1}: '{pro[i].prompt if hasattr(pro[i], 'prompt') else pro[i]}'")
            print(f" Próxima palavra prevista: '{palavra_prevista}'\n")

    path = model.get_path_to_vocab_file()
    print("\nPATH: ", path)


    print("\n===CONTEXT===")
    results = funcformer(model, func, pro)
    print(type(results))
    print(len(results))
    print(len(results[0]))


    nome = "fn_add_numbers"

    tokens = model.encode(nome)
    token_ids = tokens[0].tolist()
    print("teste:", token_ids)

    token_ids = [8822, 2891, 32964]

    for token_id in token_ids:
        print(token_id, "->", model.decode([token_id]))
