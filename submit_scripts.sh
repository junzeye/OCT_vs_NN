DIM=20
SUB="dim_$DIM"

for file in ./slurm_scripts/vary_width/*; do
    if [[ "$file" == *"$SUB"* ]]; then
        sbatch $file
    fi
done