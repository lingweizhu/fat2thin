#!/bin/sh

#SBATCH --account=rrg-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=han8@ualberta.ca
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=11:57:00
#SBATCH --constraint=cascade

module load mujoco
chmod +x scripts/tasks_*
cd $SLURM_SUBMIT_DIR/../
export OMP_NUM_THREADS=1
source $HOME/gpu_env2/bin/activate
'experiment/scripts/tasks_'"$SLURM_ARRAY_TASK_ID"'.sh'