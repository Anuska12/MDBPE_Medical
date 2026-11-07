#!/bin/bash
#SBATCH --job-name=mdbpe_acdc_4d
#SBATCH --time=6-00:00:00
#SBATCH --mem=64G             
#SBATCH --cpus-per-task=4    
#SBATCH --account=your_account
#SBATCH --output=/your/path/acdc_4d_%j.out
#SBATCH --error=/your/path/acdc_4d_%j.err

module load python/3.11.5
module load scipy-stack
source ~/mdbpe_env/bin/activate
cd ~/your/path/codes
python3 processor_acdc_mdbpe.py

