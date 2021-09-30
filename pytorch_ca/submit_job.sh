#!/bin/bash
#PBS -l nodes=1:ppn=2,mem=2gb
#PBS -l walltime=1:00:00
#PBS -N hyperparam_search

module load conda/4.9.2

cd $PBS_O_WORKDIR
cd neural_ca/pytorch_ca

conda activate gpu
wandb agent neural_ca/NeuralCA/0bavlnat

done