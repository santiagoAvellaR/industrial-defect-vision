.PHONY: install test clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

clean:
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
