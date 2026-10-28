*This project was created as part of the 42 curriculum by hohu.*

# Call Me Maybe

## Description

Call Me Maybe is a local *function calling* tool for language models. It
receives natural-language requests and a list of available functions, selects
the most likely function, and generates values for its parameters.

The goal of the project is to produce structured output without relying on
free-form text generation. It uses the local `Qwen/Qwen3-0.6B` model, provided
by the `llm_sdk` package, and restricts the tokens that can be selected during
generation.

The current output is a JSON array. Each item includes the original prompt,
the selected function name, and the generated parameters:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "fn_name": "fn_add_numbers",
  "args": {
    "a": 2,
    "b": 3
  }
}
```

## Instructions

### Requirements

- Python 3.10 or later.
- [uv](https://docs.astral.sh/uv/).
- Internet access the first time dependencies and the Hugging Face model are
  downloaded.

### Installation

The project uses `uv` and includes a `Makefile`. The `install` target creates
cache directories for packages and models, then installs runtime and
development dependencies:

```bash
make install
```

By default, the caches are configured in the `Makefile` with these variables:

```makefile
HF_HOME = /home/hohu/sgoinfre/call_me_maybe/.llm
UV_CACHE_DIR = /home/hohu/sgoinfre/call_me_maybe/.uv_cache
```

They can be overridden on the command line:

```bash
make install HF_HOME=/ruta/a/modelos UV_CACHE_DIR=/ruta/a/uv-cache
```

### Running the project

The most direct way to run the program with the files included in this
repository is:

```bash
uv run python -m src \
  --input input/function_calling_tests.json \
  --function_definitions input/functions_definition.json \
  --output output/function_calling_results.json
```

The model is downloaded or loaded while creating `LLMModel`. The first run may
take longer than later runs because of this download and the local cache.

The available arguments are:

```text
--input <file>                    JSON file containing the prompts.
--function_definitions <file>     JSON file containing the available functions.
--output <file>                   Path to the output JSON file.
```

The `Makefile` also provides these targets:

```bash
make run
make debug
make clean
make lint
make lint-strict
```

`make debug` starts the program with the standard `pdb` debugger. `make clean`
removes Python, mypy, and pytest caches.

> Note: `make run` uses the default paths configured in
> `src/parsing/arguments.py` (`data/input/...`). In this copy of the
> repository, the examples are in `input/...`; therefore, use the command with
> explicit paths shown above, or create the expected `data/input/` structure.

## Input format

The prompts file must be a JSON array of objects with a `prompt` key:

```json
[
  {"prompt": "Greet Ada"},
  {"prompt": "What is the sum of 2 and 3?"}
]
```

The definitions file must be a JSON array. Each function includes a name,
description, parameters, and return type:

```json
[
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {"type": "string"}
    },
    "returns": {"type": "string"}
  }
]
```

The parsing layer uses Pydantic models to validate the structure. In addition,
a `model_validator` verifies that every declared parameter includes a `type`
field. Missing files, insufficient permissions, invalid JSON, empty lists,
and incorrect structures are converted into explanatory errors.

## Constrained decoding algorithm

Generation is divided into two phases.

1. Function selection: the program tokenizes every valid function name. At
   each step, it checks the LLM logits and allows only the next token that
   continues at least one available name. When the generated prefix matches a
   complete name and no longer function has the same prefix, the function is
   selected.
2. Parameter generation: for each parameter of the selected function, a
   prompt containing the parameter names is created. At each step, the logits
   are sorted and the highest-scoring token whose text is allowed by a
   type-dependent mask is accepted. Generation stops when a quotation mark is
   produced or when the 64-character limit is reached. The generated text is
   then converted to the declared JSON type, including strings, integers,
   numbers, and booleans.

The model adapter obtains the public vocabulary path through
`get_path_to_vocab_file()`, loads the token-to-ID JSON once, and reverses it
into an ID-to-token mapping. Parameter constraints use this cached mapping
instead of decoding every candidate token separately.

This constraint reduces the model's search space: it cannot invent a function
name outside the definitions, and parameter values remain within a character
set compatible with their basic type.

## Design decisions

- `LLMProtocol` describes the minimum interface required by the processor:
  `encode`, `decode`, and `get_logits_from_input_ids`. This prevents the
  processor from being directly coupled to a specific SDK.
- `LLMModel` encapsulates `Small_LLM_Model`. Its Pydantic fields represent the
  model configuration, and its private `_model` attribute stores the actual
  SDK instance after initialization.
- `PromptProcessor` creates its own LLM internally and works with
  `FunctionDefinition` objects instead of untyped dictionaries.
- Each parsing module opens and validates its own file. This keeps the expected
  format and its error messages together, without a generic helper layer.
- `argparse` exposes the input and output paths through the command line.

## Performance analysis

The main cost is one LLM inference for every generated token. Consequently,
initial model loading and prompt length directly affect total runtime. The
Hugging Face cache prevents the model from being downloaded again on later
runs.

Function selection is more reliable than open generation because every token
must belong to a valid name prefix. For parameters, character masks reduce
free-text results, although they do not by themselves provide complete
semantic validation: a value may follow the allowed characters and still fail
to answer the request correctly. Exact accuracy and execution time should be
measured with a test set representative of the execution environment.

## Challenges encountered

- Integrating a local LLM without leaking its details throughout the project.
  This was addressed with the `LLMModel` adapter and `LLMProtocol`.
- Ensuring the generated name belongs to the available function set. This was
  addressed by comparing the generated tokenized prefix with every permitted
  name on each iteration.
- Handling malformed input. Pydantic models and specific exceptions were added
  for missing files, insufficient permissions, invalid JSON, and incompatible
  structures.
- Avoiding repeated downloads of the model and dependencies. The `Makefile`
  defines separate caches for Hugging Face and uv.

## Testing strategy

Project validation combines static checks and manual integration testing:

- `make lint` runs `flake8` and `mypy` with strict type rules for functions
  without annotations, `Any` returns, and unused ignores.
- `make lint-strict` runs the strict mypy variant.
- `python -m compileall src` checks that all modules compile.
- Valid input files and edge cases are tested: missing files, insufficient
  permissions, invalid JSON, empty lists, empty prompts, and parameters
  without a `type`.
- Output is reviewed as JSON and compared with the available definitions: the
  name must exist and the generated parameters must match the selected
  function.

## Usage examples

Run with explicit paths:

```bash
uv run python -m src \
  --input input/function_calling_tests.json \
  --function_definitions input/functions_definition.json \
  --output output/function_calling_results.json
```

Run the debugger:

```bash
make debug
```

Check style and types:

```bash
make lint
```

## Resources

- [Python argparse documentation](https://docs.python.org/3/library/argparse.html).
- [Pydantic documentation](https://docs.pydantic.dev/).
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html).
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/).
- [Qwen3 model documentation](https://huggingface.co/Qwen/Qwen3-0.6B).
- [Function Calling and Other API Updates](https://openai.com/index/function-calling-and-other-api-updates/), an introduction to using models to select structured tools.

### Use of AI

A conversational AI tool was used as support during development to explain
Python, Pydantic, `Protocol`, `argparse`, data validation, and code
organization concepts. It was also used to suggest refactorings, error
messages, and the first draft of this documentation. The final structural
decisions, paths, commands, current code, and checks were reviewed against the
actual repository contents.
