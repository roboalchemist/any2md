.PHONY: test test-unit test-integration test-slow test-all lint format install rust-build rust-test rust-parity rust-parity-long rust-install

# Prefer the project venv so tests spawn the interpreter that has any2md installed.
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

test:
	$(PYTHON) -m pytest tests/ -x -q --ignore=tests/test_inference.py --ignore=tests/test_integration.py

test-unit: test

test-integration:
	$(PYTHON) -m pytest tests/test_integration.py -m integration -v

test-slow:
	$(PYTHON) -m pytest tests/test_inference.py -m slow -v -s

test-all:
	$(PYTHON) -m pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

install:
	uv pip install -e '.[all]'

# --- native Rust port (ANY2-37) ---
rust-build:
	cd rust && cargo build --release

rust-test:
	cd rust && cargo test --release

# Install the native binary next to the Python CLI
rust-install: rust-build
	install -m 0755 rust/target/release/any2md-rs $${PREFIX:-/usr/local}/bin/any2md-rs

# Native vs Python parity gates (see rust/parity/parity.py)
rust-parity: rust-build
	.venv/bin/python rust/parity/parity.py mel --input tests/audio/yt_interview.wav --kind whisper
	.venv/bin/python rust/parity/parity.py mel --input tests/audio/zh_short.wav --kind nemo
	.venv/bin/python rust/parity/parity.py lid
	.venv/bin/python rust/parity/parity.py stt
	.venv/bin/python rust/parity/parity.py asr
	.venv/bin/python rust/parity/parity.py diarize
	.venv/bin/python rust/parity/parity.py reference readerlm --input tests/fixtures/sample.html \
		--max-tokens 64 --out /tmp/any2md-parity-py.json
	rust/target/release/any2md-rs html tests/fixtures/sample.html -o /tmp/any2md-parity-rust \
		--max-tokens 64 --stats-json /tmp/any2md-parity-rust.json
	.venv/bin/python rust/parity/parity.py compare /tmp/any2md-parity-py.json /tmp/any2md-parity-rust.json

# Long-audio gate (82-min podcast): bounded memory/time + permutation-invariant DER
rust-parity-long: rust-build
	.venv/bin/python rust/parity/parity.py diarize-long --stats
