PYTHON ?= python
IN ?= tickets.jsonl
OUT ?= pred.jsonl

.PHONY: predict train evaluate test run

predict:
	$(PYTHON) -m app.cli predict --in "$(IN)" --out "$(OUT)"

train:
	$(PYTHON) -m app.cli train

evaluate:
	$(PYTHON) -m app.cli evaluate --out "$(OUT)"

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

run:
	$(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

