#!/bin/bash
#SBATCH --job-name=mdbpe_pros_derma_512
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpubase_bycore_b5
#SBATCH --account=your_account
#SBATCH --output=/your/path/derma_pros_512_%j.out
#SBATCH --error=/your/path/derma_pros_512_%j.err

module load python/3.11.5
module load scipy-stack
source ~/mdbpe_env/bin/activate

cd your/path/2d_rgb
python3 processor_dermamnist_mdbpe.py



