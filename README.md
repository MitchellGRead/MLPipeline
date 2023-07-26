
# Virtual Environment
```bash
make venv
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