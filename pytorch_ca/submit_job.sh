#!/bin/bash
#PBS -l nodes=1:ppn=2,mem=2gb
#PBS -l walltime=1:00:00
#PBS -N hyperparam_search

cd $PBS_O_WORKDIR
cd neural_ca/pytorch_ca

module load conda/4.9.2
conda init bash
conda activate gpu

wandb agent neural_ca/NeuralCA/g2528j3i & wandb agent neural_ca/NeuralCA/g2528j3i

done
