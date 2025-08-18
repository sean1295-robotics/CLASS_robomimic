
#!/bin/bash

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate CLASS_robomimic

# Fine-tune the policy
python robomimic/scripts/train.py \
--dataset ./datasets/square/ph/image_v15.hdf5 \
--config robomimic/exps/templates/diffusion_policy_train.json