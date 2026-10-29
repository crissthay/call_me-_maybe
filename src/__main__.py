"""Entry point.

uv run python -m src [--functions_definition F] [--input F] [--output F]
"""
import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from llm_sdk.llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .classes import FunctionCallResult
from .decoder import build_token_sets, load_vocab, resolve_prompt
from .input_loader import read_json_func, read_json_prompt


def parse_args() -> argparse.Namespace:
    """Read the command-line arguments (with the default paths)."""
    parser = argparse.ArgumentParser(prog='call-me-maybe')
    parser.add_argument(
        '--functions_definition', type=Path,
        default=Path('data/input/functions_definition.json'),
    )
    parser.add_argument(
        '--input', type=Path,
        default=Path('data/input/function_calling_tests.json'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('data/output/function_calling_results.json'),
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

    print('Loading model (first run may download it)...', flush=True)
    try:
        model = Small_LLM_Model()
        vocab = load_vocab(model.get_path_to_vocab_file())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f'ERROR - could not load the model/vocabulary: {exc}')
    sets = build_token_sets(vocab)
    print('Model ready.', flush=True)

    results: list[FunctionCallResult] = []
    for number, entry in enumerate(prompts, start=1):
        started = time.time()
        print(f'[{number}/{len(prompts)}] {entry.prompt}', flush=True)
        try:
            results.append(
                resolve_prompt(model, vocab, sets, functions, entry.prompt)
            )
            print(f'    done in {time.time() - started:.1f}s', flush=True)
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