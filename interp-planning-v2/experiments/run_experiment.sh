#!/bin/bash
#SBATCH --job-name=interp_exp
#SBATCH --output=/scratch/gpfs/TSILVER/de7281/interp_planning/logs/job_%A_%a.out
#SBATCH --error=/scratch/gpfs/TSILVER/de7281/interp_planning/logs/job_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Experiment configuration
# 3 waypoint types (n, c, i) x 5 seeds = 15 experiments
WAYPOINT_TYPES=("n" "c" "i", "g")
SEEDS=(42 42 42)

# Get the specific configuration for this array task
WAYPOINT_TYPE=${WAYPOINT_TYPES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Load necessary modules
module load intel-mkl/2024.2  # Load Intel MKL for numpy

# Set up paths
CODE_DIR="/home/de7281/thesis/interp-planning/interp-planning/interp-planning-v2"
CONDA_ENV="interp_env"
SCRATCH_DIR="/scratch/gpfs/TSILVER/de7281/interp_planning"

# Create necessary directories
mkdir -p "$SCRATCH_DIR/logs"

# Create experiment-specific run directory
NUM_EPOCHS=10000
K=50
N=5
O=0
RUN_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR="$SCRATCH_DIR/outputs/run_${RUN_TIMESTAMP}_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Also copy logs to the run directory
exec > >(tee -a "$RUN_DIR/slurm_${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$RUN_DIR/slurm_${SLURM_JOB_ID}.err" >&2)

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Verify dependencies are available
if ! python -c "import torch; import numpy; import omegaconf" 2>/dev/null; then
    echo "ERROR: Required dependencies not found!"
    echo "Please ensure the conda environment is set up correctly."
    echo "Required packages: torch, numpy, omegaconf, hydra-core"
    exit 1
fi

echo "Dependencies verified successfully!"

# Navigate to code directory
cd "$CODE_DIR" || exit 1

# Set PYTHONPATH
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# Print configuration information
echo "========================================"
echo "Interpolative Planning Experiment"
echo "========================================"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Run Directory: $RUN_DIR"
echo "Start Time: $(date)"
echo "Submitted By: $USER"
echo "Host: $(hostname)"
echo ""
echo "Experiment Configuration:"
echo "- Waypoint Type: ${WAYPOINT_TYPE}"
echo "- Seed: ${SEED}"
echo "- Grid size (K): ${K}"
echo "- Dimensions (N): ${N}"
echo "- Obstacles (O): ${O}"
echo "- Training epochs: ${NUM_EPOCHS}"
echo ""
echo "Paths:"
echo "- Code Directory: $CODE_DIR"
echo "- Conda Environment: $CONDA_ENV"
echo ""
echo "SLURM Resources:"
echo "- CPUs: $SLURM_CPUS_PER_TASK"
echo "- Memory: 16GB"
echo "- Time Limit: 12:00:00"
echo "========================================"
echo ""

# Save run info to file
cat > "$RUN_DIR/run_info.txt" << EOF
========================================
Interpolative Planning Experiment
========================================
Job Array ID: $SLURM_ARRAY_JOB_ID
Array Task ID: $SLURM_ARRAY_TASK_ID
Job ID: $SLURM_JOB_ID
Run Directory: $RUN_DIR
Start Time: $(date)
Submitted By: $USER
Host: $(hostname)

Experiment Configuration:
- Waypoint Type: ${WAYPOINT_TYPE}
- Seed: ${SEED}
- Grid size (K): ${K}
- Dimensions (N): ${N}
- Obstacles (O): ${O}
- Training epochs: ${NUM_EPOCHS}

Paths:
- Code Directory: $CODE_DIR
- Conda Environment: $CONDA_ENV

SLURM Resources:
- CPUs: $SLURM_CPUS_PER_TASK
- Memory: 16GB
- Time Limit: 12:00:00
========================================
EOF

# Run the experiment
echo "Starting training and evaluation..."
echo ""

# Set Hydra output directory to the run directory
export HYDRA_FULL_ERROR=1

# Run with Hydra overrides (this script saves models automatically)
python experiments/run_experiment.py \
    seed=${SEED} \
    env.K=${K} \
    env.N=${N} \
    env.O=${O} \
    env.r=5 \
    env.gamma=0.99 \
    model.k=32 \
    model.hidden_dims=[128,128,64] \
    training.num_epochs=${NUM_EPOCHS} \
    training.N_T=100 \
    training.batch_size=128 \
    training.buffer_size=20000 \
    training.learn_frequency=10 \
    training.learning_rate=5e-3 \
    training.temperature=1.0 \
    training.eval_frequency=500 \
    planner.waypoint_type=${WAYPOINT_TYPE} \
    planner.max_waypoints=20 \
    planner.M=100 \
    planner.eps=0.5 \
    planner.waypoint_temp=1.0 \
    planner.num_gmm_comps=3 \
    planner.num_gmm_iters=25 \
    eval.num_test_pairs=100 \
    eval.num_trials_per_pair=5 \
    eval.max_steps=500 \
    eval.temperature=0 \
    hydra.run.dir="$RUN_DIR"

EXIT_CODE=$?

# Print completion info
echo ""
echo "========================================"
echo "Experiment completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "Output saved to: $RUN_DIR"
echo "========================================"

# List output files
echo ""
echo "Output files:"
ls -lh "$RUN_DIR"

exit $EXIT_CODE
