#!/bin/bash
#SBATCH --job-name=vit_organ
#SBATCH --time=5-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --account=your_account
#SBATCH --output=/your/path/vit_organ%j.out
#SBATCH --error=/your/path/vit_organ%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load python/3.11.5
module load scipy-stack
source ~/mdbpe_env/bin/activate
cd ~/your/path/3d
python3 -u organmnist3d_classifier.py