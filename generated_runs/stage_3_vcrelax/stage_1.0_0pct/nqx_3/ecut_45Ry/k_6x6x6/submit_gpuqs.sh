#!/bin/bash
#SBATCH --job-name=qe_%j
#SBATCH --output=qe-%j.out
#SBATCH --error=qe-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-23:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=nsfq
#SBATCH --nodelist=g[27,30]

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

module load nvidia/nvhpc/24.11
module load quantumespresso/7.4.1

mpirun -n $SLURM_NTASKS pw.x < si.opt.in > si.opt.out
