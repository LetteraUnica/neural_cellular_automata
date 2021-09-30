#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=1:00:00


cd $PBS_O_WORKDIR
cd neural_cellular_automata/pytorch_ca

conda activate gpu

python3 neural_ca_benchmark.py
