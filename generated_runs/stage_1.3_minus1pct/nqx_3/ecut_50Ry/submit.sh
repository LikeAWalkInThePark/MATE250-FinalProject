#!/bin/bash
#SBATCH --job-name=qe_%j          
#SBATCH --output=qe-%j.out
#SBATCH --error=qe-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=400G
#SBATCH --partition=nsfqs
#SBATCH --nodelist=g[27,30]

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

#export OMPI_MCA_btl=self,tcp
#export OMPI_MCA_btl_tcp_if_exclude=lo,docker0
#export OMPI_MCA_btl_openib_allow_ib=0

module load nvidia/nvhpc/24.11 
module load quantumespresso/7.5.0

mpirun -n $SLURM_NTASKS pw.x -npool 2 -ndiag 2 < si.scf.in > si.scf.out

