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
export EXPERIMENT_NAME="llm"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'

python pipeline/train.py \
    --model-to-train "Tagifai_LLM_Model" \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "./data/labeled_projects.csv" \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-workers 2 \
    --cpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-loc results/
```

# Tuning from CLI
```bash
export EXPERIMENT_NAME="llm"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
export INITIAL_PARAMS="[{\"train_loop_config\": $TRAIN_LOOP_CONFIG}]"

python pipeline/tune.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "./data/labeled_projects.csv" \
    --initial-params "$INITIAL_PARAMS" \
    --num-runs 2 \
    --num-workers 2 \
    --cpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-loc results/
```
