#!/bin/bash
#SBATCH --time=180
#SBATCH --job-name=rcnn
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wilsonwi@comp.nus.edu.sg
#SBATCH --gpus=h100-96:1
#SBATCH --mem=0
#SBATCH --output=output_rcnn.log
#SBATCH --error=error_rcnn.log
#SBATCH --partition=gpu

srun train.sh