#!/bin/bash
#PBS -l nodes=1:ppn=4,mem=4gb
#PBS -l walltime=1:00:00
#PBS -N hyperparam_search

cd $PBS_O_WORKDIR
cd neural_ca/pytorch_ca
git pull

module load conda/4.9.2
conda init bash
conda activate gpu


#for i in $N_GPU #metti la variabile giusta per il numero di gpu
#do 
#    CUDA_VISIBLE_DEVICES=$i, wandb agent neural_ca/NeuralCA/2t8zwc8y &

CUDA_VISIBLE_DEVICES=0, wandb agent neural_ca/mask/9wzcztye & CUDA_VISIBLE_DEVICES=1, wandb agent neural_ca/mask/9wzcztye

done
