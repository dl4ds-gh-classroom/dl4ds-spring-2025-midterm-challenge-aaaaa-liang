#!/bin/bash
#$ -P dl4ds                   # Use the correct project name
#$ -N simple_cnn_training     # Job name
#$ -l h_rt=01:00:00           # Increased time limit (adjust if needed)
#$ -m e                        # Send an email when job ends
#$ -M yuchiliang15@gmail.com   # Non-BU email
#$ -j y                        # Merge error and output files
#$ -pe omp 8                   # Increased to 8 CPU cores (adjust if needed)
#$ -l mem_per_core=4G          # Reduced per-core memory to stay within limits
#$ -l gpus=1                   # Request 1 GPU
#$ -l gpu_c=6.0                # Minimum GPU compute capability
#$ -q academic-gpu             # **Use the correct SCC GPU queue**

# Activate virtual environment
source /projectnb/dl4ds/students/annliang/dl4ds-spring-2025-midterm-challenge-aaaaa-liang/.venv/bin/activate

# Navigate to project directory
cd /projectnb/dl4ds/students/annliang/dl4ds-spring-2025-midterm-challenge-aaaaa-liang

# Run **starter_code.py** instead of my_cnn_script.py
python3 starter_code.py > /projectnb/dl4ds/students/annliang/job_output.log 2>&1

# Move CSV output to project directory (ensure it's not saved in the home directory)
mv submission_ood.csv /projectnb/dl4ds/students/annliang/submission_ood.csv
