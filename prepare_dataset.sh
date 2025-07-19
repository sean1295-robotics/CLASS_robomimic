
#!/bin/bash

# Download dataset for square task
python robomimic/scripts/download_datasets.py \
--tasks square \
--dataset_types ph \
--hdf5_types low_dim

# Convert dataset states to observations with camera images
python robomimic/scripts/dataset_states_to_obs.py \
--dataset robomimic/datasets/square/ph/low_dim_v15.hdf5 \
--output_name robomimic/datasets/square/ph/image_v15.hdf5 \
--camera_names agentview \
--camera_height 256 \
--camera_width 256

# Add absolute actions to the dataset
python robomimic/scripts/conversion/robosuite_add_absolute_actions.py \
--dataset robomimic/datasets/square/ph/image_v15.hdf5

# Extract action dictionary from the dataset
python robomimic/scripts/extract_action_dict.py \
--dataset robomimic/datasets/square/ph/image_v15.hdf5