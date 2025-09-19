#!/usr/bin/env bash
#
#SBATCH --job-name=CA-MTL-YOLOv8
#SBATCH --time=01:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               
#SBATCH --constraint=dgx           
#SBATCH --error=log_CA-MTL-YOLOv8.%J.err
#SBATCH --output=log_CA-MTL-YOLOv8.%J.out

# ===============================
# User-configurable paths
# ===============================

HYPERPARAMETERS_YAML="/mnt/home/users/tic_163_uma/mpascual/execs/CAMTL_YOLO/hyperparams/defaults.yaml"

# Print job information
echo "====================================="
echo "SLURM JOB: CA-MTL-YOLOv8"
echo "====================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $PWD"
echo "====================================="
echo


module purge
module load miniconda
source activate camtl_yolo

# Execute the experiment
python camtl_yolo.train --cfg "$CONFIG_FILE"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo
    echo "=================================================="
    echo "CA-MTL-YOLOv8 experiment completed successfully!"
    echo "=================================================="
    echo "Results saved in (local default): $RESULTS_DIR/"
    echo "Note: If output_root is set in YAML, results are under that root:"
    echo "  - Model checkpoints: $RESULTS_DIR/gen_diffusive_*.pth"
    echo "  - Training samples: $RESULTS_DIR/sample_discrete_epoch_*.png"
    echo "  - Validation metrics: $RESULTS_DIR/val_*.npy"
    echo "  - Test results: $RESULTS_DIR/generated_samples/"
else
    echo
    echo "========================================="
    echo "ERROR: CA-MTL-YOLOv8 experiment failed!"
    echo "========================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check error logs: log_CA-MTL-YOLOv8.$SLURM_JOB_ID.err"
    exit $EXIT_CODE
fi

echo "End Time: $(date)"
echo "Total GPU time: $(($(date +%s) - $(date -d "$SLURM_JOB_START_TIME" +%s))) seconds"