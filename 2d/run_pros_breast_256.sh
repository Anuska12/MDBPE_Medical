#!/bin/bash
#SBATCH --job-name=mdbpe_processor
#SBATCH --time=5-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpubase_bycore_b5
#SBATCH --account=your_account
#SBATCH --output=/your/account/processor_%j.out
#SBATCH --error=/your/account/processor_%j.err

module load python/3.11.5
module load scipy-stack
source ~/mdbpe_env/bin/activate

cd ~/your/path/2d
python3 processor_breastmnist_mdbpe.py


