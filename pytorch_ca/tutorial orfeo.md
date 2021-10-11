# Tutorial orfeo

## Comandi base
`ssh s277227@ct1-005.area.trieste.it` connettersi a orfeo  
`qstat` Controllare la coda, -Q meno informazioni, -n più informazioni  
`pbsnodes -aS` Controllare le risorse dei nodi  
`qsub -q "queue name" "bash script"` Submit a job  
`qsub -q dssc -l nodes=1:ppn=4,walltime=00:10:00 -I` Interactive job with 4 processor and 1 node for 10 minutes on dssc queue  
`lscpu` Check current cpu  
`nvidia-smi` Check GPU status  
`fuser -v /dev/nvidia*` List processes executing on GPU  
`kill -9 "PID"` Kill process number PID  
