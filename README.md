# ML Pipeline
A Machine Learning automation pipeline

# Virtual Environment
```bash
make venv
```

# Dev Setup
```bash
make dev
```

# Docs Setup
```bash
make docs
python3 -m mkdocs new .
python3 -m mkdocs serve
```


# Training from CLI
```bash
python tagifai/train.py \
    --experiment-name "test_experiment" \
    --dataset-loc ./data/dataset.csv \
    --num-workers 5 \
    --cpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/training_results.json
```
