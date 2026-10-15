# MonoGeoFlightLog developer entry points.
#
# All targets are thin wrappers around the commands documented in the README so
# there is exactly one way to do each thing.

PY ?= .venv/bin/python
CONFIG ?= configs/default.yaml
# Directory containing the C++ OpenCV installation. On macOS a project-local
# conda env is enough: conda create -p .cv -c conda-forge opencv
OPENCV_DIR ?= $(CURDIR)/.cv/lib/cmake/opencv4
BUILD_DIR ?= cpp/build

.PHONY: help install venv test lint fmt run clean cpp cpp-configure cpp-test report figures

help:
	@echo "make install        install the package (pip install -e .) with dev extras"
	@echo "make venv           create .venv and install everything"
	@echo "make test           pytest (Python) -- the C++ tests are 'make cpp-test'"
	@echo "make lint           ruff check + black --check"
	@echo "make fmt            ruff --fix + black (rewrites files)"
	@echo "make run            full pipeline: bash scripts/reproduce.sh"
	@echo "make cpp            configure + build + ctest the C++17 inference path"
	@echo "make cpp-test       ctest only"
	@echo "make figures        rebuild results/figures from results/*.json"
	@echo "make report         rebuild results/RESULTS.md from results/*.json"
	@echo "make clean          remove build trees and cached runs (keeps results/)"

install:
	$(PY) -m pip install -e ".[dev]"

venv:
	python3.11 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m black --check src tests scripts

fmt:
	$(PY) -m ruff check src tests scripts --fix
	$(PY) -m black src tests scripts

run:
	CONFIG=$(CONFIG) bash scripts/reproduce.sh

figures:
	$(PY) scripts/make_plots.py --results results --config $(CONFIG)

report:
	$(PY) -m monogeo.cli report --config $(CONFIG) --out results/RESULTS.md

cpp-configure:
	cmake -S cpp -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$(OPENCV_DIR)

cpp: cpp-configure
	cmake --build $(BUILD_DIR) -j
	ctest --test-dir $(BUILD_DIR) --output-on-failure

cpp-test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf cpp/build build dist src/monogeo.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf runs data/cache
	@echo "kept results/ (run 'rm -rf results' if you really want those gone)"
