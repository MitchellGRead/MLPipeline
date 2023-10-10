#!/bin/bash
#SBATCH --mem=125G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:3
#SBATCH --time=14:0:0
#SBATCH --mail-user=<stevenbobyn@gmail.com>
#SBATCH --mail-type=ALL

cd ~/$projects/MLPipeline
module purge
module load python/3.11
source ./pipeline-venv/bin/activate

python pipeline/train.py mesh-train-model \
    --dataset-loc "../meshgraphnets/MeshGraphNets_PyG/datasets/meshgraphnets_miniset30traj5ts_vis.pt" \
    --train-loop-config "./config/mesh_graph_net.json" \
    --num-workers 2 \
    --batch-size 2
