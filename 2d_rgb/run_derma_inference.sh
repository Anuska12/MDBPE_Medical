#!/bin/bash
#SBATCH --account=your_account
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=derma_inf
#SBATCH --output=/your/path_%j.out

module load python          
source ~/mdbpe_env/bin/activate
cd /your/path/2d_rgb
python3 -u derma_inference_time.py