import argparse
import json
import sys
from pathlib import Path
from pydantic import ValidationError
from llm_sdk.llm_sdk import Small_LLM_Model  #type: ignore[attr-defined] documentacaooo
from .classes import FunctionCallResult
from .decoder import build_token_sets, load_vocab, resolve_prompt
from .input_loader import read_json_func, read_json_prompt

def parse_args() -> argparse.Namespace:
    """Read the command-line arguments (with the default paths)."""
    parser = argparse.ArgumentParser(prog='call-me-maybe')
    parser.add_argument(
        '--functions_definition', type=Path,
        default=Path('datas/functions_definition.json'),
    )
    parser.add_argument(
        '--input', type=Path,
        default=Path('datas/function_calling_tests.json'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('datas/function_calling_results.json'),
    )
    return parser.parse_args()


def main() -> None:
    """Load inputs, run the model on every prompt, write the output."""
    args = parse_args()

    try:
        functions = read_json_func(args.functions_definition)
        prompts = read_json_prompt(args.input)
    except FileNotFoundError as exc:
        sys.exit(f'ERROR - file not found: {exc.filename}')
    except json.JSONDecodeError as exc:
        sys.exit(f'ERROR - invalid JSON: {exc}')
    except ValidationError as exc:
        sys.exit(f'ERROR - invalid input structure:\n{exc}')

    if not functions:
        sys.exit('ERROR - no functions defined in the definition file')

    try:
        model = Small_LLM_Model()
        vocab = load_vocab(model.get_path_to_vocab_file())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f'ERROR - could not load the model/vocabulary: {exc}')
    sets = build_token_sets(vocab)

    results: list[FunctionCallResult] = []
    for entry in prompts:
        try:
            results.append(
                resolve_prompt(model, vocab, sets, functions, entry.prompt)
            )
        except (ValueError, IndexError, RuntimeError) as exc:
            print(f'WARNING - skipped "{entry.prompt}": {exc}',
                  file=sys.stderr)

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as file:
            json.dump([r.model_dump() for r in results], file, indent=2)
    except OSError as exc:
        sys.exit(f'ERROR - could not write output: {exc}')

    print(f'Done: {len(results)}/{len(prompts)} prompts -> {args.output}')

if __name__ == '__main__':
    main()


#test main
"""if __name__ == "__main__":
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
            print(type(logits_da_frase))
            print(type(logits_da_frase[0]))
            print(logits_da_frase[0])
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
        print(token_id, "->", model.decode([token_id]))"""
