UV         := $(HOME)/.local/bin/uv
SGOINFRE   := $(shell [ -d /sgoinfre/$(USER) ] && echo /sgoinfre/$(USER) || echo $(HOME))
VENV_DIR   := $(SGOINFRE)/.venv_cmm

UV_ENV     := UV_CACHE_DIR=$(SGOINFRE)/.cache/uv \
              UV_PROJECT_ENVIRONMENT=$(VENV_DIR) \
              HF_HOME=$(SGOINFRE)/.cache/huggingface

UV_RUN     := $(UV_ENV) $(UV) run

.PHONY: install run debug clean lint lint-strict

install:
	mkdir -p $(SGOINFRE)/.cache/uv $(SGOINFRE)/.cache/huggingface $(VENV_DIR)
	$(UV_ENV) $(UV) sync

run:
	$(UV_RUN) python -m src

debug:
	$(UV_RUN) python -m pdb -m src

lint:
	$(UV_RUN) flake8 .
	$(UV_RUN) mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV_RUN) flake8 .
	$(UV_RUN) mypy . --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf data/output/*
