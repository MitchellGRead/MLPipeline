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
