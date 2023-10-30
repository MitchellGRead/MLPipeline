#!/bin/bash
#SBATCH --mem=125G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:1
#SBATCH --time=14:0:0
#SBATCH --mail-user=<stevenbobyn@gmail.com>
#SBATCH --mail-type=ALL

cd $def/MLPipeline
module purge
module load python/3.11
source ./pipeline-venv/bin/activate

dataset=$1

python pipeline/train.py gns-train-model \
    --dataset-loc "./data/Mpm88Small/$dataset" \
    --train-loop-config "./config/complex_physics_gns.json" \
    --num-workers 2 \
    --batch-size 2
