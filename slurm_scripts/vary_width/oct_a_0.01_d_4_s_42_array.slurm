#!/bin/bash
#SBATCH --job-name=oct_a0.01_d4_s42_array
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)

#SBATCH --array=20,40,60,80,100,120%1 # use hidden_dim as task ID
#SBATCH --output=../../synthetic_tests/CTG_width/out/oct_a0.01_d4_s42_%a.out    # Standard output and error log
#SBATCH --time=3:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins, but not for end.
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=junzey@princeton.edu

module purge
module load anaconda3/2022.5 gurobi/9.5.1
conda activate thesis

cd ../../
python experiment_instance.py --a 0.01 --d 4 --s 42 --hidden_dim SLURM_ARRAY_TASK_ID