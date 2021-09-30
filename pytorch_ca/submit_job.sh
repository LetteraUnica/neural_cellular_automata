#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -l walltime=1:00:00


cd $PBS_O_WORKDIR
cd neural_ca/pytorch_ca

conda activate gpu
wandb agent neural_ca/NeuralCA/0bavlnat

done