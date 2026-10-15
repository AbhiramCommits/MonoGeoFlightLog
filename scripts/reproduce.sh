#!/usr/bin/env bash
#
# End-to-end reproduction of every number in results/.
#
# Usage:
#   bash scripts/reproduce.sh              # full pipeline
#   SKIP_CPP=1 bash scripts/reproduce.sh   # skip the CMake/C++ benchmark
#
# Everything is written to results/ and the whole run is teed to
# results/run_log.txt so the provenance of each number is auditable.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="${PY:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

CONFIG="${CONFIG:-configs/default.yaml}"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

# Everything below is teed to the run log.
exec > >(tee "$RESULTS/run_log.txt") 2>&1

banner() {
  echo
  echo "=============================================================="
  echo "== $*"
  echo "=============================================================="
}

echo "MonoGeoFlightLog reproduction run"
echo "date:       $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "root:       $ROOT"
echo "python:     $($PY --version 2>&1) ($PY)"
echo "config:     $CONFIG"
echo "host:       $(uname -srm)"
echo "cpu count:  $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')"

banner "1/9 camera calibration from checkerboards"
$PY -m monogeo.cli calibrate --config "$CONFIG" --out results/calibration.json

banner "2/9 classical two-view baseline (frame gap 2, plus gap 1 reference)"
$PY -m monogeo.cli two-view --config "$CONFIG" --sequence forward --out results/two_view_baseline.json
$PY -m monogeo.cli two-view --config "$CONFIG" --sequence forward --gap 1 --out results/two_view_baseline_gap1.json

banner "3/9 self-supervised depth + ego-motion training"
$PY -m monogeo.cli train-depth --config "$CONFIG"

banner "4/9 two-view pointmap training"
$PY -m monogeo.cli train-pointmap --config "$CONFIG"

banner "5/9 three-way comparison with scale analysis"
$PY -m monogeo.cli compare --config "$CONFIG" --out results/method_comparison.json

banner "6/9 clean held-out metrics"
$PY -m monogeo.cli evaluate --config "$CONFIG" --out results/metrics.json

banner "7/9 corruption + domain-shift stress tests"
$PY -m monogeo.cli stress --config "$CONFIG" --out results/stress_tests.json

banner "8/9 figures"
$PY scripts/make_plots.py --results results --config "$CONFIG"

banner "9/9 C++17 ONNX inference, parity check and benchmark"
if [ "${SKIP_CPP:-0}" = "1" ]; then
  echo "SKIP_CPP=1 set: skipping ONNX export and C++ build"
else
  $PY -m monogeo.cli export --config "$CONFIG" --out artifacts/depth_net.onnx
  cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build cpp/build -j
  ctest --test-dir cpp/build --output-on-failure
  $PY scripts/bench_cpp_vs_python.py
fi

banner "report"
$PY -m monogeo.cli report --config "$CONFIG" --out results/RESULTS.md

echo
echo "Done. Artifacts:"
ls -1 "$RESULTS" | sed 's/^/  results\//'
echo
echo "Full log: results/run_log.txt"
