#!/bin/bash
#SBATCH --time=180
#SBATCH --job-name=vitmae
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wilsonwi@comp.nus.edu.sg
#SBATCH --gpus=h100-47:4
#SBATCH --mem=0
#SBATCH --output=output_vitmae.log
#SBATCH --error=error_vitmae.log
#SBATCH --partition=gpu

srun models/vitmae/train.sh