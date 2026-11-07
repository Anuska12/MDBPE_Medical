#!/bin/bash
#SBATCH --job-name=vit_acdc
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:1
#SBATCH --account=your_account
#SBATCH --output=/your/path/vit_acdc_%j.out
#SBATCH --error=/your/path/vit_acdc_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load python/3.11.5
module load scipy-stack
source ~/mdbpe_env/bin/activate

cd ~/your/path/codes/4d
python3 -u acdc_classifier.py