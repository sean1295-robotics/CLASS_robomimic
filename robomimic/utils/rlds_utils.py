"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict

import tensorflow as tf
import torch
import tensorflow_graphics.geometry.transformation as tfg
import numpy as np

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