#!/bin/bash
#$ -P dl4ds                   # Use the correct project name
#$ -N sophisticated_cnn_training  # Updated job name
#$ -l h_rt=01:00:00           # Increased time limit (adjust if needed)
#$ -m e                        # Send an email when job ends
#$ -M yuchiliang15@gmail.com   # Non-BU email
#$ -j y                        # Merge error and output files
#$ -pe omp 8                   # Increased to 8 CPU cores (adjust if needed)
#$ -l mem_per_core=4G          # Reduced per-core memory to stay within limits
#$ -l gpus=1                   # Request 1 GPU
#$ -l gpu_c=6.0                # Minimum GPU compute capability
#$ -q academic-gpu             # **Use the correct SCC GPU queue**

# Limit thread usage to prevent SCC from over-allocating CPU resources
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export TBB_NUM_THREADS=8

# Activate virtual environment
source /projectnb/dl4ds/students/annliang/dl4ds-spring-2025-midterm-challenge-aaaaa-liang/.venv/bin/activate

# Navigate to project directory
cd /projectnb/dl4ds/students/annliang/dl4ds-spring-2025-midterm-challenge-aaaaa-liang

# Run **sophisticated_cnn.py** and redirect output to log file with "_soph_cnn"
python3 sophisticated_cnn.py > /projectnb/dl4ds/students/annliang/job_output_soph_cnn.log 2>&1

# Move CSV output to project directory with "_soph_cnn"
mv submission_ood.csv /projectnb/dl4ds/students/annliang/submission_ood_soph_cnn.csv
