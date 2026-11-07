#!/bin/bash
#SBATCH --job-name=mdbpe_pros_organ3d_512
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=your_account
#SBATCH --output=/your/path/pros_organ3d_512_%j.out
#SBATCH --error=/your/path/pros_organ3d_512_%j.err

module load python/3.11.5
module load scipy-stack
source ~/mdbpe_env/bin/activate

cd ~/your/path
python3 -u processor_organmnist3d_mdbpe.py

echo "Job finished"
date