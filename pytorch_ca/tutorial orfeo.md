# Tutorial orfeo

## Comandi base
`ssh s277227@ct1-005.area.trieste.it` connettersi a orfeo  
`qstat` Controllare la coda, -Q meno informazioni, -n più informazioni  
`pbsnodes -aS` Controllare le risorse dei nodi  
`qsub -q "queue name" "bash script"` Submit a job  
`qsub -q dssc -l nodes=1:ppn=48,walltime=00:30:00 -I` Interactive job with 48 processor, 2 GPUs and 1 node for 30 minutes on dssc queue  
`lscpu` Check current cpu  
`nvidia-smi` Check GPU status  
`fuser -v /dev/nvidia*` List processes executing on GPU  
`kill -9 "PID"` Kill process number PID  
