"""
This script converts robosuite dataset with delta actions to absolute actions.
It reads the robomimic dataset, processes each demo to convert delta actions (`actions`) to 
absolute actions, and saves the results back to the dataset under new keys:
- `actions_abs_pose`: absolute pose actions (position + orientation + gripper) - 10D
- `actions_abs_joint`: absolute joint actions (joint positions + gripper) - 8D

Arguments:
    dataset (str): path to the robomimic dataset
    num_workers (int): number of workers to use for parallel processing

Example usage:
    python scripts/conversion/robosuite_add_absolute_actions.py --dataset /path/to/your/demo.hdf5 --num_workers 10
"""

import multiprocessing
import pathlib
import h5py
from tqdm import tqdm
import argparse
import numpy as np
import copy
import torch

import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from scipy.spatial.transform import Rotation

import robosuite

from robomimic.config import config_factory




"""
copied/adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/common/robomimic_util.py
"""
class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        """
        Class to convert robomimic dataset with delta actions to absolute actions.
        Args:
            dataset_path (str): path to the robomimic dataset
            algo_name (str): name of the algorithm to use for config
        """
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        
        # Create absolute pose environment
        abs_pose_env_meta = copy.deepcopy(env_meta)
        if robosuite.__version__ < "1.5":
            abs_pose_env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        else:
            abs_pose_env_meta['env_kwargs']['controller_configs']['body_parts']['right']['input_type'] = "absolute"
        
        # Create absolute joint environment
        abs_joint_env_meta = copy.deepcopy(env_meta)
        if robosuite.__version__ < "1.5":
            abs_joint_env_meta['env_kwargs']['controller_configs']['type'] = 'JOINT_POSITION'
            abs_joint_env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            abs_joint_env_meta['env_kwargs']['controller_configs']['input_max'] = np.pi
            abs_joint_env_meta['env_kwargs']['controller_configs']['input_min'] = -np.pi
            abs_joint_env_meta['env_kwargs']['controller_configs']['output_max'] = np.pi
            abs_joint_env_meta['env_kwargs']['controller_configs']['output_min'] = -np.pi
        else:
            abs_joint_env_meta['env_kwargs']['controller_configs']['body_parts']['right']['type'] = 'JOINT_POSITION'
            abs_joint_env_meta['env_kwargs']['controller_configs']['body_parts']['right']['input_type'] = "absolute"
            abs_joint_env_meta['env_kwargs']['controller_configs']['body_parts']['right']['input_max'] = np.pi
            abs_joint_env_meta['env_kwargs']['controller_configs']['body_parts']['right']['input_min'] = -np.pi
            abs_joint_env_meta['env_kwargs']['controller_configs']['body_parts']['right']['output_max'] = np.pi
            abs_joint_env_meta['env_kwargs']['controller_configs']['body_parts']['right']['output_min'] = -np.pi

        # Create environments
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert len(env.env.robots) in (1, 2)

        abs_pose_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_pose_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        
        abs_joint_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_joint_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        self.env = env
        self.abs_pose_env = abs_pose_env
        self.abs_joint_env = abs_joint_env
        self.file = h5py.File(dataset_path, 'r')
    
    def get_demo_keys(self):
        return list(self.file['data'].keys())

    def convert_actions_pose(self, 
            states: np.ndarray, 
            actions: np.ndarray,
            initial_state: dict) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        env = self.env
        d_a = len(env.env.robots[0].action_limits[0])

        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, d_a)

        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_remainder = stacked_actions[...,6:]
        
        for i in range(len(states)):
            if i == 0:
                _ = env.reset_to(initial_state)
            else:
                _ = env.reset_to({'states': states[i]})

            for idx, robot in enumerate(env.env.robots):
                if robosuite.__version__ < "1.5":     
                    # read pos and ori from robots
                    controller = robot.controller                
                else:
                    # read pos and ori from robots
                    controller = robot.part_controllers['right']
                    
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(
                    controller.goal_ori).as_rotvec()

        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_remainder
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_actions_joint(self, 
            states: np.ndarray, 
            actions: np.ndarray,
            initial_state: dict) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent joint positions for each step
        keep the original gripper action intact.
        """
        env = self.env
        joint_env = self.abs_joint_env
        d_a = len(env.env.robots[0].action_limits[0])
        
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, d_a)
        
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_qpos = np.zeros(
            stacked_actions.shape[:-1]+(7,), 
            dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]
        
        for i in range(len(states)):
            if i == 0:
                _ = env.reset_to(initial_state)
                _ = joint_env.reset_to(initial_state)
            else:
                _ = env.reset_to({'states': states[i]})
                _ = joint_env.reset_to({'states': states[i]})
            
            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
                joint_robot = joint_env.env.robots[idx]
                
                # read pos and ori from robots
                if robosuite.__version__ < "1.5":  
                    controller = robot.controller
                    joint_controller = joint_robot.controller
                else:
                    controller = robot.part_controllers['right']
                    joint_controller = joint_robot.part_controllers['right']
                
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(
                    controller.goal_ori).as_rotvec()
                
                torques = controller.torques
                desired_torque = np.linalg.solve(joint_controller.mass_matrix, torques - joint_controller.torque_compensation)
                joint_pos = np.array(controller.sim.data.qpos[controller.qpos_index])
                joint_vel = np.array(controller.sim.data.qvel[controller.qvel_index])
                position_error = (desired_torque + np.multiply(joint_vel, joint_controller.kd)) / joint_controller.kp
                desired_qpos = position_error + joint_pos
                action_goal_qpos[i,idx] = desired_qpos
                
        action_goal_pos = torch.from_numpy(action_goal_pos)
        action_goal_ori = torch.from_numpy(action_goal_ori)
        action_goal_qpos = torch.from_numpy(action_goal_qpos)
        action_gripper = torch.from_numpy(action_gripper)
        
        # For pose output (10D per robot: 3 pos + 3 ori + 1 gripper)
        stacked_abs_actions_pose = torch.cat([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], dim=-1)
        abs_actions_pose = stacked_abs_actions_pose.reshape(-1, 7 * len(env.env.robots))
        
        # For joint output (8D per robot: 7 joint + 1 gripper)
        stacked_abs_actions_joint = torch.cat([
            action_goal_qpos,
            action_gripper
        ], dim=-1)
        abs_actions_joint = stacked_abs_actions_joint.reshape(-1, 8 * len(env.env.robots))
        
        return abs_actions_pose.numpy(), abs_actions_joint.numpy()

    def convert_demo(self, demo_key):
        file = self.file
        demo = file["data/{}".format(demo_key)]
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo.attrs["model_file"]
        initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)

        # generate abs pose actions (original method)
        abs_actions_pose_old = self.convert_actions_pose(states, actions, initial_state=initial_state)
        
        # generate abs pose and joint actions (new method)
        abs_actions_pose_new, abs_actions_joint = self.convert_actions_joint(states, actions, initial_state=initial_state)
        
        return abs_actions_pose_old, abs_actions_pose_new, abs_actions_joint


"""
copied/adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/scripts/robomimic_dataset_conversion.py
"""
def worker(x):
    path, demo_key = x
    converter = RobomimicAbsoluteActionConverter(path)
    abs_actions_pose_old, abs_actions_pose_new, abs_actions_joint = converter.convert_demo(demo_key)
    info = dict()
    return abs_actions_pose_old, abs_actions_pose_new, abs_actions_joint, info


def add_absolute_actions_to_dataset(dataset, num_workers):
    """
    Entry-point for adding absolute actions to robomimic dataset.
    Args:
        dataset (str): path to the robomimic dataset
        num_workers (int): number of workers to use for parallel processing
    """
    # process inputs
    dataset = pathlib.Path(dataset).expanduser()
    assert dataset.is_file()

    # initialize converter
    converter = RobomimicAbsoluteActionConverter(dataset)
    demo_keys = converter.get_demo_keys()
    del converter
    
    # run
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker, [(dataset, demo_key) for demo_key in demo_keys])

    # modify action
    with h5py.File(dataset, 'r+') as out_file:
        for i in tqdm(range(len(results)), desc="Writing to output"):
            abs_actions_pose_old, abs_actions_pose_new, abs_actions_joint, info = results[i]
            demo = out_file["data/{}".format(demo_keys[i])]
            
            # Original pose actions (backward compatibility)
            if "actions_abs" not in demo:
                demo.create_dataset("actions_abs", data=np.array(abs_actions_pose_old))
            else:
                demo['actions_abs'][:] = abs_actions_pose_old
            
            # New pose actions (7D per robot: 3 pos + 3 ori + 1 gripper)
            if "actions_abs_pose" not in demo:
                demo.create_dataset("actions_abs_pose", data=np.array(abs_actions_pose_new))
            else:
                demo['actions_abs_pose'][:] = abs_actions_pose_new
                
            # Joint actions (8D per robot: 7 joint + 1 gripper)
            if "actions_abs_joint" not in demo:
                demo.create_dataset("actions_abs_joint", data=np.array(abs_actions_joint))
            else:
                demo['actions_abs_joint'][:] = abs_actions_joint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
    )
    
    args = parser.parse_args()

    add_absolute_actions_to_dataset(
        dataset=args.dataset,
        num_workers=args.num_workers,
    )