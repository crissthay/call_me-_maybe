import os
from .input_loader import read_json_func, read_json_prompt, promptformer
from llm_sdk.llm_sdk.__init__ import Small_LLM_Model


if __name__ == "__main__":
    pastaatual = os.path.dirname(os.path.abspath(__file__))
    path_json = os.path.abspath(
        os.path.join(pastaatual, "..", "datas", "functions_definition.json")
    )
    path_prompt = os.path.abspath(
        os.path.join(pastaatual, "..", "datas", "function_calling_tests.json")
    )
    func = read_json_func(path_json)
    func1 = read_json_prompt(path_prompt)
    
    print("JSON:\n")
    for line in func:
        print("Name =", line.name)
        print("Description =", line.description)
        print("Parameters =", line.parameters)
        print("Returns =", line.returns)
       

    print("\nPROMPT:\n")
    for line in func1:
        print(line)
    

    print("\n === TEST encode ====")
    model = Small_LLM_Model()
    res = promptformer(model, func1)
    print(res)