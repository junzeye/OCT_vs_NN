#!/bin/bash
#SBATCH --job-name=oct_a0.001_d6_s53_array
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=6               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)

#SBATCH --array=20,40,60,80,100,120%1 # use hidden_dim as task ID
#SBATCH --output=../../synthetic_tests/CTG_width/out/oct_a0.001_d6_s53_%a.out    # Standard output and error log
#SBATCH --time=3:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=junzey@princeton.edu

module purge
module load anaconda3/2022.5 gurobi/9.5.1
conda activate thesis

cd ../../
python experiment_instance.py --a 0.001 --d 6 --s 53 --hidden_dim SLURM_ARRAY_TASK_ID