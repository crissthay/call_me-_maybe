NAME    = olha vejo depois
VENV    = venv
PYTHON  = $(VENV)/bin/python
PIP     = $(VENV)/bin/pip

.SILENT:

all: install

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

build: install
	$(PYTHON) -m build --wheel --outdir .

run: install
	$(PYTHON) $(NAME) $(CONFIG)

debug: install
	$(PYTHON) -m pdb $(NAME) $(CONFIG)

lint: install
	$(VENV)/bin/flake8 . --exclude venv
	$(VENV)/bin/mypy . --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

clean:
	rm -rf venv build dist *.egg-info
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	rm -rf venv

.PHONY: all install build run debug lint lint-strict clean