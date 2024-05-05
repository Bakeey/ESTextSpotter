#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpumem:20g
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Load necessary modules
module load gcc/8.2.0 python/3.8.5 cuda/11.3.1

# Activate your virtual environment if needed
# source /path/to/your/virtualenv/bin/activate

# Install necessary Python modules
pip install torch numpy click pycocotools

# List of directories containing pickles
directories=("hb_gimbal_long_1_better" "hb_gimbal_long_2" "hb_gimbal_long_3" "hb_gimbal_long_4" "hb_gimbal_short_1_worse" "hb_gimbal_short_2")

# Loop through each directory and run the Python script
for dir in "${directories[@]}"; do
    input_dir="/cluster/home/kiten/images/hb_iphone_stable_bf/text_detections/$dir"
    output_dir="/cluster/home/kiten/images/hb_iphone_stable_bf/text_detections_cpu/$dir"  # Adjust this path as needed

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Run the Python script
    python deserialize_pickle.py --input_dir "$input_dir" --output_dir "$output_dir"
done
