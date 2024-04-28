#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpumem:20g
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cd /cluster/work/riner/users/PLR-2024/kiten/ESTextSpotter

# Set Matplotlib backend to 'Agg' (non-interactive backend suitable for scripts that do not require a GUI)
# Environment variable for Matplotlib
export MPLBACKEND='Agg'

# Ensure .py is executable
chmod +x ./vis.py
chmod +x ./detect.py

# Step 7: Go to ops directory
cd ./models/ests/ops

# Step 9: Compile MultiScaleDeformableAttention (This must run on a GPU node)
python3 setup.py build develop --prefix=/cluster/home/kiten/.local

# Step 10: Run vis script
cd /cluster/work/riner/users/PLR-2024/kiten/ESTextSpotter
python3 detect.py --model_config_path "./config/ESTS/ESTS_5scale_ctw1500_finetune.py" \
                 --model_checkpoint_path "/cluster/home/kiten/totaltext_checkpoint.pth" \
                 --image_dir "/cluster/home/kiten/images/test" \
                 --out_dir "/cluster/home/kiten/output/test/text_detections" \
                 --out_dir_vis "/cluster/home/kiten/output/test/images"
python3 detect.py --model_config_path "./config/ESTS/ESTS_5scale_ctw1500_finetune.py" \
                 --model_checkpoint_path "/cluster/home/kiten/totaltext_checkpoint.pth" \
                 --image_dir "/cluster/home/kiten/images/pics" \
                 --out_dir "/cluster/home/kiten/output/pics/text_detections" \
                 --out_dir_vis "/cluster/home/kiten/output/pics/images"