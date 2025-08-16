"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict

import tensorflow as tf
import torch
import tensorflow_graphics.geometry.transformation as tfg
import numpy as np
import hashlib


def filter_success(trajectory: dict[str, any]):
    # only keep trajectories that have "success" in the file path
    return tf.strings.regex_full_match(
        trajectory['traj_metadata']['episode_metadata']['file_path'][0],
        ".*/success/.*"
    )

def euler_to_quat(euler):
    return tfg.quaternion.from_euler(euler)
    
def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)

def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat

def get_coordinate_transform_matrix():
    """
    Get the transformation matrix to convert from DROID coordinate frame 
    to the target coordinate frame.
    
    Simplified transformation matrix (fully rounded since rotations are random samples).
    """
    # Simplified transformation matrix
    R_relative = tf.constant([[ 0.0,  -1.0,  0.0],
                             [1.0,  0.0,  0.0],
                             [ 0.0,  0.0,  1.0]], dtype=tf.float32)
    
    return R_relative

def apply_rotation_transform(rotation_matrix, transform_matrix):
    """
    Apply coordinate transformation to rotation matrices.
    
    Args:
        rotation_matrix: [..., 3, 3] rotation matrices
        transform_matrix: [3, 3] transformation matrix
    
    Returns:
        transformed rotation matrices: transform_matrix @ rotation_matrix
    """
    # Ensure both matrices have the same dtype
    rotation_matrix = tf.cast(rotation_matrix, tf.float32)
    transform_matrix = tf.cast(transform_matrix, tf.float32)
    
    # Expand transform_matrix to match batch dimensions
    batch_shape = tf.shape(rotation_matrix)[:-2]
    transform_expanded = tf.broadcast_to(transform_matrix, 
                                       tf.concat([batch_shape, [3, 3]], axis=0))
    
    # Apply transformation: R_new = R_relative @ R_original
    transformed_matrix = tf.linalg.matmul(transform_expanded, rotation_matrix)
    
    return transformed_matrix

def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform DROID dataset to be consistent with target coordinate frame.
    """
    # Extract translation and cast to float32
    T = tf.cast(trajectory["action_dict"]["cartesian_position"][:, :3], tf.float32)
    
    # Extract rotation and convert to rotation matrix
    euler_angles = trajectory["action_dict"]["cartesian_position"][:, 3:6]
    rotation_matrix = euler_to_rmat(euler_angles)
    
    # Apply coordinate frame transformation
    transform_matrix = get_coordinate_transform_matrix()
    transformed_rotation_matrix = apply_rotation_transform(rotation_matrix, transform_matrix)
    
    # Convert back to 6D representation
    R = mat_to_rot6d(transformed_rotation_matrix)
    
    # Cast gripper position to float32
    gripper = tf.cast(trajectory["action_dict"]["gripper_position"], tf.float32)
    
    # Combine everything
    trajectory["action"] = tf.concat(
        (
            T,
            R,
            gripper,
        ),
        axis=-1,
    )
    return trajectory

def robomimic_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform DROID dataset to be consistent with target coordinate frame.
    """
    eef_pos = tf.cast(trajectory["observation"]["cartesian_position"][:, :3], tf.float32) + tf.constant([-0.56, 0.0, 0.912])
    eef_quat = euler_to_quat(trajectory["observation"]["cartesian_position"][:, 3:6])  
    signs = tf.sign(eef_quat[..., 0:1])  # Get sign of first element
    eef_quat = eef_quat * signs  # Multiply entire quaternion by sign
    gripper = 0.04 *  tf.concat([
                -trajectory["observation"]["gripper_position"][..., -1:],  # First element: negative
                trajectory["observation"]["gripper_position"][..., -1:]    # Second element: original
            ], axis=-1)
    trajectory["observation"].update({
        "robot0_eef_pos": eef_pos,
        "robot0_eef_quat": eef_quat,
        "robot0_gripper_qpos": gripper,
    })    
    # Extract translation and cast to float32
    T = tf.cast(trajectory["action_dict"]["cartesian_position"][:, :3], tf.float32) + tf.constant([-0.56, 0.0, 0.912]) #robomimic offset
    # Extract rotation and convert to rotation matrix
    euler_angles = trajectory["action_dict"]["cartesian_position"][:, 3:6]
    rotation_matrix = euler_to_rmat(euler_angles)
    
    # Apply coordinate frame transformation
    transform_matrix = get_coordinate_transform_matrix()
    transformed_rotation_matrix = apply_rotation_transform(rotation_matrix, transform_matrix)
    
    # Convert back to 6D representation
    R = mat_to_rot6d(transformed_rotation_matrix)
    
    # Cast gripper position to float32
    gripper_action = 2 * tf.cast(trajectory["action_dict"]["gripper_position"], tf.float32) - 1   #robomimic gripper is in [-1, 1] range  
    # Combine everything
    trajectory["action"] = tf.concat(
        (
            T,
            R,
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory

def robomimic_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "obs": {
            "agentview_image":  trajectory["observation"]["image_primary"],
            "robot0_eye_in_hand_image": trajectory["observation"]["image_secondary"],
            "lang_emb": trajectory["task"]["language_instruction"],
            "robot0_eef_pos": trajectory["observation"]["proprio"][:, :3],
            "robot0_eef_quat": trajectory["observation"]["proprio"][:, 3:7],
            "robot0_gripper_qpos": trajectory["observation"]["proprio"][:, 7:],
        },
        "actions": trajectory["action"][1:],
    }
    
def roboarena_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"].update(dict(
        arm_joint_pos= trajectory["observation"]["joint_position"],
        gripper_pos= trajectory["observation"]["gripper_position"],
    ))    
    joint_action = trajectory["action_dict"]["joint_position"]    
    gripper_action = trajectory["action_dict"]["gripper_position"]
    
    # Combine everything
    trajectory["action"] = tf.concat(
        (
            joint_action,
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory

def roboarena_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    table_cam = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: trajectory["observation"]["image_primary1"],
                lambda: trajectory["observation"]["image_primary2"],
            )
    print(trajectory.keys())
    print(trajectory["observation"].keys())
    instruction = tf.random.shuffle(
                [trajectory["task"]["language_instruction"], trajectory["task"]["language_instruction_2"], trajectory["task"]["language_instruction_3"]]
            )[0]
    return {
        "obs": {
            "table_cam": table_cam,
            "wrist_cam": trajectory["observation"]["image_secondary"],
            "lang_emb": instruction,
            "arm_joint_pos": trajectory["observation"]["proprio"][..., :7],
            "gripper_pos": trajectory["observation"]["proprio"][..., -1:],
        },
        "actions": trajectory["action"][1:],
    }
    
DROID_TO_RLDS_OBS_KEY_MAP = {
    "agentview_image": "exterior_image_1_left",
    "robot0_eye_in_hand_image": "exterior_image_2_left"
}

DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP = {
    "robot_state/cartesian_position": "cartesian_position",
    "robot_state/gripper_position": "gripper_position",
}

class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)
        
        
        
        
import os
import numpy as np
import tensorflow as tf
import torch
from tqdm.auto import tqdm
from typing import Dict, Any
import json
import torchvision.transforms as T
import random
import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds

class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        droid_name: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        action_chunk_size: int = 16,
        action_space: bool = True,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        normalize: bool = True,
    ):
        
        tf.config.set_visible_devices([], "GPU")

        builder = tfds.builder(droid_name, data_dir=data_dir)
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )
        dataset = dataset.repeat()

        def restructure(traj):
            actions = tf.concat(
                (
                    traj["action_dict"]["joint_position"],
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            exterior_img = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]
            
            return {
                "actions": actions,
                "obs": {
                    "table_cam": exterior_img,
                    "wrist_cam": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                    "lang_emb": instruction,
                },
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)           
        self.proprio_keys = [k for k in dataset.element_spec["obs"].keys() if "cam" not in k and "rgb" not in k and "lang" not in k]

        if normalize:
            self.normalization_stats = self._get_normalization_stats(
                data_dir,
                droid_name,
                action_space,
            )
            self._convert_stats_to_tensors()
            dataset = dataset.traj_map(self._normalize_fn, num_parallel_calls)
        else:
            self.normalization_stats = None

        def chunk_actions(traj):
            traj_len = tf.shape(traj["actions"])[0]
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        def filter_idle(traj):
            return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)

        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(filter_idle)

        def decode_and_unsqueeze_obs(traj):
            traj["obs"]["table_cam"] = tf.io.decode_image(
                traj["obs"]["table_cam"], expand_animations=False, dtype=tf.uint8
            )
            traj["obs"]["wrist_cam"] = tf.io.decode_image(
                traj["obs"]["wrist_cam"], expand_animations=False, dtype=tf.uint8
            )
            for k in self.proprio_keys:
                traj["obs"][k] = tf.expand_dims(traj["obs"][k], axis=0)
            traj["obs"]["table_cam"] = tf.expand_dims(traj["obs"]["table_cam"], axis=0)
            traj["obs"]["wrist_cam"] = tf.expand_dims(traj["obs"]["wrist_cam"], axis=0)
            
            return traj

        dataset = dataset.frame_map(decode_and_unsqueeze_obs, num_parallel_calls)   
        
        # Define the augmentation function
        def augmentations(traj):
            
            # Define the configuration for the augmentations
            augmentation_config = {
                "augment_order": ["random_resized_crop", "random_brightness", "random_contrast", "random_saturation", "random_hue"],
                "random_resized_crop": {
                    "scale": [0.8, 1.0],
                    "ratio": [0.9, 1.1],
                },
                "random_brightness": [0.3],
                "random_contrast": [0.6, 1.4],
                "random_saturation": [0.5, 1.5],
                "random_hue": [0.3],
            }

            # Define a helper function to apply all augmentations to a single image
            def apply_all_augmentations(img):
                # Convert to float for color-based augmentations
                img = tf.image.convert_image_dtype(img, tf.float32)
                # Squeeze the image to remove the batch dimension (from [1, H, W, C] to [H, W, C])
                img = tf.squeeze(img, axis=0)
                for aug_name in augmentation_config["augment_order"]:
                    if aug_name == "random_resized_crop":
                        scale_min, scale_max = augmentation_config["random_resized_crop"]["scale"]
                        ratio_min, ratio_max = augmentation_config["random_resized_crop"]["ratio"]
                        
                        # Get the original image dimensions
                        original_height = tf.shape(img)[0]
                        original_width = tf.shape(img)[1]
                        
                        # Randomly determine the crop area and aspect ratio
                        area_factor = tf.random.uniform([], minval=scale_min, maxval=scale_max)
                        target_area = tf.cast(original_height * original_width, tf.float32) * area_factor
                        
                        log_ratio_min = tf.math.log(ratio_min)
                        log_ratio_max = tf.math.log(ratio_max)
                        aspect_ratio = tf.math.exp(tf.random.uniform([], minval=log_ratio_min, maxval=log_ratio_max))
                        
                        # Calculate new crop dimensions
                        height = tf.clip_by_value(tf.cast(tf.sqrt(target_area * aspect_ratio), tf.int32), 1, original_height)
                        width = tf.clip_by_value(tf.cast(tf.sqrt(target_area / aspect_ratio), tf.int32), 1, original_width)
                        
                        # Apply the random crop
                        img = tf.image.random_crop(img, size=[height, width, 3])
                        
                        # Resize the cropped image back to the original size
                        img = tf.image.resize(img, [original_height, original_width])

                    elif aug_name == "random_brightness":
                        max_delta = augmentation_config["random_brightness"][0]
                        img = tf.image.random_brightness(img, max_delta=max_delta)
                    
                    elif aug_name == "random_contrast":
                        lower, upper = augmentation_config["random_contrast"]
                        img = tf.image.random_contrast(img, lower=lower, upper=upper)
                        
                    elif aug_name == "random_saturation":
                        lower, upper = augmentation_config["random_saturation"]
                        img = tf.image.random_saturation(img, lower=lower, upper=upper)

                    elif aug_name == "random_hue":
                        max_delta = augmentation_config["random_hue"][0]
                        img = tf.image.random_hue(img, max_delta=max_delta)

                # Convert back to uint8 to match the rest of the pipeline
                return tf.image.convert_image_dtype(img, tf.uint8)

            # Apply the augmentations to the 'table_cam' and 'wrist_cam'
            traj["obs"]["table_cam"] = apply_all_augmentations(traj["obs"]["table_cam"])[None]
            traj["obs"]["wrist_cam"] = apply_all_augmentations(traj["obs"]["wrist_cam"])[None]
            
            return traj

        dataset = dataset.frame_map(augmentations, num_parallel_calls)  

        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def _convert_stats_to_tensors(self):
        """Converts normalization stats from Python lists to TensorFlow tensors."""
        if self.normalization_stats is None:
            return

        for k in self.normalization_stats["obs"]:
            if self.normalization_stats["obs"][k] is not None:
                self.normalization_stats["obs"][k]["mean"] = tf.convert_to_tensor(self.normalization_stats["obs"][k]["mean"], dtype=tf.float64)
                self.normalization_stats["obs"][k]["std"] = tf.convert_to_tensor(self.normalization_stats["obs"][k]["std"], dtype=tf.float64)

        if "actions" in self.normalization_stats:
            if "min" in self.normalization_stats["actions"] and "max" in self.normalization_stats["actions"]:
                self.normalization_stats["actions"]["min"] = tf.convert_to_tensor(self.normalization_stats["actions"]["min"], dtype=tf.float64)
                self.normalization_stats["actions"]["max"] = tf.convert_to_tensor(self.normalization_stats["actions"]["max"], dtype=tf.float64)
            self.normalization_stats["actions"]["mean"] = tf.convert_to_tensor(self.normalization_stats["actions"]["mean"], dtype=tf.float64)
            self.normalization_stats["actions"]["std"] = tf.convert_to_tensor(self.normalization_stats["actions"]["std"], dtype=tf.float64)
            
    def _normalize_fn(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        stats = self.normalization_stats
        
        # Min-Max Normalization based on percentiles for actions
        if "actions" in traj and "actions" in stats and "min" in stats["actions"] and "max" in stats["actions"]:
            actions = traj["actions"]
            actions_min = stats["actions"]["min"]
            actions_max = stats["actions"]["max"]
            
            denom = actions_max - actions_min
            denom = tf.where(tf.abs(denom) < 1e-8, tf.constant(1.0, dtype=tf.float64), denom)
            
            traj["actions"] = 2.0 * (actions - actions_min) / denom - 1.0
            # traj["actions"] = tf.clip_by_value(traj["actions"], -1.0, 1.0)

        return traj

    def __iter__(self):
        for batch in self.dataset.as_numpy_iterator():
            new_batch = {
                "actions": torch.from_numpy(batch["actions"]),
                "obs": {
                    "table_cam": torch.from_numpy(batch["obs"]["table_cam"]),
                    "wrist_cam": torch.from_numpy(batch["obs"]["wrist_cam"]),
                    "lang_emb": batch["obs"]["lang_emb"],
                }
            }
            for k in self.proprio_keys:
                new_batch["obs"][k] = torch.from_numpy(batch["obs"][k])
            
            yield new_batch

    def __len__(self):
        return 20_000_000

    def _get_normalization_stats(self, data_dir: str, droid_name: str, action_space: bool) -> Dict[str, Any]:
        """Loads/computes and caches normalization stats based on dataset profile."""
        profile_string = f"{data_dir}_{droid_name}_{action_space}"
        profile_hash = hashlib.sha256(profile_string.encode('utf-8')).hexdigest()
        
        stats_dir = os.path.join(data_dir, droid_name)
        stats_path = os.path.join(stats_dir, f"normalization_stats_{profile_hash}.json")
        
        if os.path.exists(stats_path):
            print(f"Found normalization stats for this dataset profile. Loading from {stats_path}...")
            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                return stats
            except (json.JSONDecodeError, KeyError):
                print("Cached stats file is invalid or corrupted. Recomputing...")

        print("Normalization stats not found or do not match dataset profile. Computing and saving them...")
        
        os.makedirs(stats_dir, exist_ok=True)
        
        # The fix is to return the full metadata dictionary which contains "actions" and "obs"
        stats = self._compute_normalization_stats(data_dir, droid_name, action_space)
        
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        print(f"Normalization stats saved to {stats_path}")
        return stats

    @staticmethod
    def _compute_normalization_stats(data_dir: str, droid_name: str, action_space: bool) -> Dict[str, Any]:
        """
        Computes the normalization statistics for a dataset, using 1st and 99th percentiles for min/max.
        """
        import dlimp as dl
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        builder = tfds.builder(droid_name, data_dir=data_dir)
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=-1)

        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )
        
        # Get all observation keys from the dataset spec
        all_obs_keys = list(dataset.element_spec["observation"].keys())
        # Determine which keys need normalization
        proprio_keys_to_normalize = [
            k for k in all_obs_keys 
            if "cam" not in k and "image" not in k and "rgb" not in k and "lang" not in k
        ]
        
        def restructure_for_stats(traj):
            actions = tf.concat(
                (
                    traj["action_dict"]["joint_position"] if action_space else traj["action_dict"]["joint_velocity"],
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            all_obs_dict = {k: traj["observation"][k] for k in all_obs_keys}
            return {"actions": actions, **all_obs_dict}

        dataset = dataset.traj_map(restructure_for_stats, num_parallel_calls=-1)

        actions = []
        obs_data = {k: [] for k in all_obs_keys}

        for traj in tqdm(dataset.iterator(), desc="Computing stats"):
            actions.append(traj["actions"])
            for k in proprio_keys_to_normalize:
                if k in traj:
                    obs_data[k].append(traj[k])

        actions = np.concatenate(actions)

        obs_stats = {}
        # Compute stats only for proprioceptive data
        for k in proprio_keys_to_normalize:
            if k in obs_data and len(obs_data[k]) > 0:
                data = np.concatenate(obs_data[k])
                obs_stats[k] = {
                    "mean": data.mean(0).tolist(),
                    "std": data.std(0).tolist(),
                }

        # Add placeholders for other keys to prevent AssertionError
        for k in all_obs_keys:
            if k not in obs_stats:
                obs_stats[k] = None

        p1 = np.percentile(actions, 1, axis=0)
        p99 = np.percentile(actions, 99, axis=0)

        metadata = {
            "actions": {
                "mean": actions.mean(0).tolist(),
                "std": actions.std(0).tolist(),
                "min": p1.tolist(),
                "max": p99.tolist(),
            },
            "obs": obs_stats,
        }
        return metadata
    
from collections import OrderedDict
def get_robomimic_action_stats(stats: Dict[str, Any]) -> OrderedDict:
    """
    Computes and returns action normalization stats in a Robomimic-style format
    (OrderedDict with scale and offset as NumPy arrays).
    """
    actions_stats = stats.get("actions", {})
    if not actions_stats:
        raise KeyError("Action normalization stats not found in the dictionary.")
        
    mean = np.array(actions_stats["mean"], dtype=np.float32)
    std = np.array(actions_stats["std"], dtype=np.float32)

    # Use a conservative range of +/- 2 standard deviations to estimate min/max
    min_val = mean - 2 * std
    max_val = mean + 2 * std

    # Calculate scale and offset
    scale = (max_val - min_val) / 2.0
    offset = (max_val + min_val) / 2.0
    
    # Handle the case where std is close to zero (e.g., for binary values)
    scale[scale < 1e-6] = 1.0
    
    return OrderedDict([
        ('actions', {
            'scale': scale[None],
            'offset': offset[None],
        })
    ])


def get_robomimic_obs_stats(stats: Dict[str, Any], select_keys: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dictionary with the mean/std for each proprioception key,
    formatted as NumPy arrays.
    """
    obs_stats_dict = {}
    for k, v in stats['obs'].items():
        if v is not None:
            obs_stats_dict[k] = {
                'offset': np.array(v['mean'], dtype=np.float32)[None],
                'scale': np.array(v['std'], dtype=np.float32)[None],
            }

    for k in select_keys:
        if k not in obs_stats_dict:
            obs_stats_dict[k] = None
            
    if not obs_stats_dict:
        raise KeyError("Proprioception normalization stats not found in the dictionary.")

    return obs_stats_dict