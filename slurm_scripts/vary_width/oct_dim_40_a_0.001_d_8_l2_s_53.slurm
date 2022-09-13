#!/bin/bash
#SBATCH --job-name=oct_dim40_a0.001_d8_l2_s53
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=18G         # memory per cpu-core (4G is default)

#SBATCH --output=./synthetic_tests/CTG_width/oct_dim40_a0.001_d8_l2_s53.out    # Standard output and error log
#SBATCH --time=03:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=junzey@princeton.edu

module purge
module load anaconda3/2022.5 gurobi/9.5.1
conda activate thesis

python experiment_instance.py --a 0.001 --d 8 --s 53 --hidden_dim 40 --min_samples_split 2 --timelimit 13800