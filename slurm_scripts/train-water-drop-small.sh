#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=10:0:0
#SBATCH --mail-user=<stevenbobyn@gmail.com>
#SBATCH --mail-type=ALL

cd ~/$projects/MLPipeline
module purge
module load python/3.11
source ./pipeline-env/bin/activate

python pipeline/train.py gns-train-model \
    --dataset-loc "../224w-gns/Datasets/WaterDropSmall" \
    --train-loop-config "./config/complex_physics_gns.json" \
    --num-workers 4
