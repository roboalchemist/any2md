.PHONY: test test-unit test-integration test-slow test-all lint format install

test:
	python -m pytest tests/ -x -q --ignore=tests/test_inference.py --ignore=tests/test_integration.py

test-unit: test

test-integration:
	python -m pytest tests/test_integration.py -m integration -v

test-slow:
	python -m pytest tests/test_inference.py -m slow -v -s

test-all:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

install:
	uv pip install -e '.[all]'
