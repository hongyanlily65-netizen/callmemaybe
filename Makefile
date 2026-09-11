HF_HOME ?= /home/hohu/sgoinfre/call_me_maybe/.llm
UV_CACHE_DIR ?= /home/hohu/sgoinfre/call_me_maybe/.uv_cache

export HF_HOME
export UV_CACHE_DIR

PYTHON ?= .venv/bin/python
UV ?= uv

.PHONY: install run debug clean lint lint-strict

install:
	mkdir -p "$(HF_HOME)" "$(UV_CACHE_DIR)"
	UV_CACHE_DIR="$(UV_CACHE_DIR)" $(UV) sync --extra dev

run:
	mkdir -p "$(HF_HOME)"
	$(PYTHON) -m src

debug:
	mkdir -p "$(HF_HOME)"
	$(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(PYTHON) -m flake8 .
	$(PYTHON) -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(PYTHON) -m flake8 .
	$(PYTHON) -m mypy . --strict
