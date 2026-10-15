# MonoGeoFlightLog

Monocular geometry for aerial / driving flight logs: classical camera physics next to learned
geometry, with a C++ deployment path.

> **Status:** the full README (architecture, run instructions and measured results) is written in
> step 7 of the build. Until then this file is a stub.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
monogeo --help
```
