# ML Pipeline
A Machine Learning automation pipeline

# Virtual Environment - Do if you only want to run a model or hit endpoints
```bash
make venv
```
Create or open the .env file and add the W&B key WEIGHT_AND_BIASES_API_KEY=...

Login to Weights & Biases
```bash
wandb login
```

# Dev Setup - Do if you only want to perform development within the repo
```bash
make dev
```
Create or open the .env file and add the W&B key WEIGHT_AND_BIASES_API_KEY=...

Login to Weights & Biases
```bash
wandb login
```

# Docs Setup
```bash
make docs
python3 -m mkdocs new .
python3 -m mkdocs serve
```

# Training from CLI
```bash
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'

python pipeline/train.py \
    --model-to-train "Tagifai_LLM_Model" \
    --dataset-loc "./data/labeled_projects.csv" \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-workers 2 \
    --cpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --eval-model \
    --results-loc results/
```

You can also provide the hyper parameters as a filepath to a json file:
```bash
python pipeline/train.py \
    --model-to-train "Tagifai_LLM_Model" \
    --dataset-loc "labeled_projects.csv:latest" \
    --train-loop-config "./config/tagifai_args.json" \
    --num-workers 2 \
    --cpu-per-worker 2 \
    --num-epochs 2 \
    --batch-size 256 \
    --results-loc results/
```

# Tuning from CLI (WIP)
```bash
DEPRECATED
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

# Evaluation from CLI
Get a run id based on some metric and save it as a variable. Here we grab the run with the lowest val_loss:
```bash
export RUN_ID=$(python pipeline/evaluate.py get-best-run-id \
    --model-to-eval "Tagifai_LLM_Model" \
    --metric "val_loss" \
    --sort-mode ASC)
```

You can run your evaluation metrics defined by the models MetricHandler with the evaluate endpoint.
This endpoint also allows you to specify a run_id or a metric to sort for similar to get-best-run-id:
```bash
python pipeline/evaluate.py evaluate \
    --model-to-eval "Tagifai_LLM_Model" \
    --dataset-loc "./data/labeled_projects.csv" \
    --results-loc results/
```

python pipeline/artifacts.py process-dataset \
    --dataset-loc "./data/labeled_projects.csv" \
    --data-type "raw_data" \
    --data-for-model-id "Tagifai_LLM_Model"

python pipeline/artifacts.py process-dataset \
    --dataset-loc "./data/test" \
    --data-type "raw_data" \
    --data-for-model-id "Tagifai_LLM_Model"
