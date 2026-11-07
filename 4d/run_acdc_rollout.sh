#!/bin/bash
#SBATCH --job-name=acdc_rollout
#SBATCH --account=your_account
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=acdc_rollout_%j.out
module load python
source ~/mdbpe_env/bin/activate
cd ~/your/path/codes/4d/k_fold
python3 -u attention_rollout_acdc.py all
python3 -u attention_rollout_acdc.py summary
python3 -u rescue_failure_acdc.py 2048 --no-fig
python3 -u rescue_failure_acdc.py 1024 --no-fig
python3 -u rescue_failure_acdc.py 2048
