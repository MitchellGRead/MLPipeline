# ML Pipeline
MLPipeline is a modular machine learning framework that integrates with Weights & Biases to manage the full experiment lifecycle—dataset versioning, model training, hyperparameter tuning, evaluation, and artifact storage. It supports graph-based physics simulations (Graph Neural Simulator and Mesh Graph Net via PyTorch Geometric) as well as NLP text classification (BERT-based Tagifai model), with distributed training through Ray. The pipeline provides a CLI interface for uploading versioned datasets as W&B artifacts, launching training runs with automatic metric logging and checkpoint management, querying best-performing runs for evaluation, and generating visualizations of model predictions. It is designed for scalable research workflows, including HPC cluster execution via SLURM scripts.

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

# Training
```bash
python pipeline/train.py gns-train-model \
    --dataset-loc "./data/complex_physics/WaterDropSmall" \
    --train-loop-config "./config/complex_physics_gns.json" \
    --num-workers 2
```

```bash
python pipeline/train.py mesh-train-model \
    --dataset-loc "./data/mesh_graph_net/meshgraphnets_miniset5traj_vis.pt" \
    --train-loop-config "./config/mesh_graph_net.json" \
    --num-workers 2
```

# Data Artifact Processing
```bash
python pipeline/artifacts.py process-dataset \
    --dataset-loc "./data/labeled_projects.csv" \
    --data-type "raw_data" \
    --data-for-model-id "Tagifai_LLM_Model"

python pipeline/artifacts.py process-dataset \
    --dataset-loc "./data/test" \
    --data-type "raw_data" \
    --data-for-model-id "Tagifai_LLM_Model"
```
