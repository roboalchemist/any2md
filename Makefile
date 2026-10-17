.PHONY: test test-unit test-integration test-slow test-all lint format install rust-build rust-test rust-parity

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

# --- native Rust port (ANY2-37) ---
rust-build:
	cd rust && cargo build --release

rust-test:
	cd rust && cargo test --release

# Native vs Python parity gates (see rust/parity/parity.py)
rust-parity: rust-build
	.venv/bin/python rust/parity/parity.py mel --input tests/audio/yt_interview.wav --kind whisper
	.venv/bin/python rust/parity/parity.py mel --input tests/audio/zh_short.wav --kind nemo
	.venv/bin/python rust/parity/parity.py lid
	.venv/bin/python rust/parity/parity.py stt
	.venv/bin/python rust/parity/parity.py asr
	.venv/bin/python rust/parity/parity.py reference readerlm --input tests/fixtures/sample.html \
		--max-tokens 64 --out /tmp/any2md-parity-py.json
	rust/target/release/any2md-rs html tests/fixtures/sample.html -o /tmp/any2md-parity-rust \
		--max-tokens 64 --stats-json /tmp/any2md-parity-rust.json
	.venv/bin/python rust/parity/parity.py compare /tmp/any2md-parity-py.json /tmp/any2md-parity-rust.json
