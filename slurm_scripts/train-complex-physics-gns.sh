#!/bin/bash
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=14:0:0
#SBATCH --mail-user=<stevenbobyn@gmail.com>
#SBATCH --mail-type=ALL

cd ~/$projects/MLPipeline
module purge
module load python/3.11
source ./pipeline-venv/bin/activate

python pipeline/train.py \
    --dataset-loc "../224w-gns/Datasets/WaterDrop" \
    --train-loop-config "./config/complex_physics_gns.json" \
    --num-workers 4 \
    --batch-size 2
