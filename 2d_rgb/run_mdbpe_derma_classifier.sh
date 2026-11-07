#!/bin/bash
#SBATCH --job-name=vit_derma
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --account=your_account
#SBATCH --output=/your/path/vit_derma%j.out
#SBATCH --error=/your/path/vit_derma%j.err

module load python/3.11.5
module load scipy-stack
source ~/mdbpe_env/bin/activate
cd ~/your/path/2d_rgb
python3 dermamnist_classifier.py