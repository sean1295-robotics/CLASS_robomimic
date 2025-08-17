
#!/bin/bash

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate CLASS_robomimic

# Precompute if DTW distances are not cached and then subsequently pretrain
python robomimic/scripts/train.py \
--dataset ./datasets/square/ph/image_v15.hdf5 \
--config robomimic/exps/templates/diffusion_policy_pretrain.json \
--pretrain