#!/bin/bash
#SBATCH --time=4320
#SBATCH --job-name=cnn_lstm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wilsonwi@comp.nus.edu.sg
#SBATCH --gpus=h100-47:2
#SBATCH --mem=0
#SBATCH --output=output_cnn_lstm.log
#SBATCH --error=error_cnn_lstm.log
#SBATCH --partition=gpu-long

srun train.sh
