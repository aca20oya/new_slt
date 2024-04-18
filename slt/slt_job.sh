#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --mail-user=oyiabdelmageed1@sheffield.ac.uk
#SBATCH --job-name=slt_job
#SBATCH --output=output.txt
#SBATCH --time=4-00:00:00 
module load Anaconda3/2022.05
module load cuDNN/8.7.0.84-CUDA-11.8.0
conda activate myenv
python -m signjoey train configs/dgs_sign.yaml
