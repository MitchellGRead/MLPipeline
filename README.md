
# Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .
```

# Dev Setup
```bash
python3 -m pip install -e ".[dev]"
```

# Docs Setup
```bash
python3 -m pip install -e ".[docs]"
python3 -m mkdocs new .
python3 -m mkdocs serve
```