"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
os.environ["MUJOCO_GL"] = "egl"
import shutil
import psutil
import sys
import traceback

from collections import OrderedDict
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import tensorflow as tf

import robosuite
import robomimic
import robomimic.utils.action_utils as ActionUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.dataset import action_stats_to_normalization_stats
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.rlds_utils import robomimic_transform, robomimic_dataset_transform, roboarena_transform, roboarena_dataset_transform, TorchRLDSDataset
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings

from octo.data.dataset import make_dataset_from_rlds, make_interleaved_dataset
from octo.data.utils.data_utils import get_dataset_statistics, combine_dataset_statistics
from octo.utils.spec import ModuleSpec

def train(config, device, resume=False):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    torch.set_num_threads(2)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config, resume=resume)

    # path for latest model and backup (to support @resume functionality)
    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)
    obs_normalization_stats = None        

    ds_format = config.train.data_format

    # FOR RLDS
    tf.config.set_visible_devices([], "GPU")
    obs_modalities = config.observation.modalities.obs.rgb
    # NOTE: Must be 2 cam for now, can clean this up later
    assert(len(obs_modalities) == 2)
    BASE_DATASET_KWARGS = {
            "data_dir": config.train.data_path,
            "image_obs_keys": {"primary1": "exterior_image_1_left", "primary2": "exterior_image_2_left",  "secondary": "wrist_image_left"},
            "state_obs_keys": ["arm_joint_pos", "gripper_pos"], #["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            "language_key": ["language_instruction", "language_instruction_2", "language_instruction_3"],
            "norm_skip_keys":  ["proprio"],
            "action_proprio_normalization_type": "bounds",
            "absolute_action_mask": [True] * 8,
            "action_normalization_mask": [config.train.normalize_action] * 8,
            "standardize_fn": roboarena_dataset_transform,
        }
    dataset_names = config.train.dataset_names
    filter_functions = [[ModuleSpec.create(
                            "robomimic.utils.rlds_utils:filter_success"
                            )] if  "droid" in d_name else [] \
                        for d_name in dataset_names]
    dataset_kwargs_list = [
        {"name": d_name, 
            "filter_functions": f_functions, 
            **BASE_DATASET_KWARGS} for d_name, f_functions in zip(dataset_names, filter_functions)
    ]
    
    # Compute combined normalization stats
    combined_dataset_statistics = combine_dataset_statistics(
        [make_dataset_from_rlds(**dataset_kwargs, train=True)[1] for dataset_kwargs in dataset_kwargs_list]
    )
    
    dataset_kwargs = {
        "dataset_kwargs_list": dataset_kwargs_list,
        "sample_weights": config.train.sample_weights,
        "train": True,
        "shuffle_buffer_size": config.train.shuffle_buffer_size,
        "batch_size": None,  # batching will be handled in PyTorch Dataloader object
        "balance_weights": False,
        "dataset_statistics": combined_dataset_statistics,
        "traj_transform_kwargs": {
            "window_size": config.algo.horizon.observation_horizon,
            "future_action_window_size": config.algo.horizon.prediction_horizon,
            "subsample_length": config.train.subsample_length,
            "skip_unlabeled": False,    # to skip all trajectories without language
        },
        "frame_transform_kwargs": {
            "image_augment_kwargs": {
                "augment_order": ["random_resized_crop", "random_brightness", "random_contrast", "random_saturation"],
                "random_resized_crop": {
                    "scale": [0.8, 1.0],
                    "ratio": [0.9, 1.1],
                },
                "random_brightness": [0.3],
                "random_contrast": [0.7, 1.3],
                "random_saturation": [0.7, 1.3],
                "random_hue": [0.3],
            },
            "num_parallel_calls": config.train.num_parallel_calls,
        },
        "traj_transform_threads": config.train.traj_transform_threads,
        "traj_read_threads": config.train.traj_read_threads,
    }
    
    init_dataset_kwargs = deepcopy(dataset_kwargs)
    init_dataset_kwargs["shuffle_buffer_size"] = 1
    
    init_trainset = make_interleaved_dataset(**init_dataset_kwargs)        
    init_trainset = init_trainset.map(roboarena_transform, num_parallel_calls=config.train.traj_transform_threads)
    action_normalization_stats = None
    
    if config.train.normalize_action:
        action_stats = ActionUtils.get_action_stats_dict(combined_dataset_statistics["action"], config.train.action_keys, config.train.action_shapes)
        action_normalization_stats = action_stats_to_normalization_stats(action_stats, config.train.action_config)
        print(action_normalization_stats)     
        
    pytorch_init_dataset = TorchRLDSDataset(init_trainset)
    init_train_loader = DataLoader(
        pytorch_init_dataset,
        batch_size=1,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )
    init_data_loader_iter = iter(init_train_loader)
    batch = next(init_data_loader_iter) 
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=None,
        batch=batch,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=True,
    )
    print("shape_meta", shape_meta)

    trainset = make_interleaved_dataset(**dataset_kwargs)        
    trainset = trainset.map(roboarena_transform, num_parallel_calls=config.train.traj_transform_threads)
    action_normalization_stats = None
    
    if config.train.normalize_action:
        action_stats = ActionUtils.get_action_stats_dict(combined_dataset_statistics["action"], config.train.action_keys, config.train.action_shapes)
        action_normalization_stats = action_stats_to_normalization_stats(action_stats, config.train.action_config)
        print(action_normalization_stats)     
        
    pytorch_dataset = TorchRLDSDataset(trainset)
    train_loader = DataLoader(
        pytorch_dataset,
        batch_size=config.train.batch_size,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )
    # add info to optim_params
    with config.values_unlocked():
        # number of learning steps per epoch (defaults to a full dataset pass)
        config.experiment.epoch_every_n_steps = int(0.9 * config.train.shuffle_buffer_size / config.train.batch_size) #each epoch is a 80% pass through the buffer
        train_num_steps = config.experiment.epoch_every_n_steps # arbitrary value   
        if "optim_params" in config.algo:
            # add info to optim_params of each net
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = train_num_steps
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs                    

    assert config.experiment.epoch_every_n_steps is not None and config.train.num_epochs is not None, "epoch_every_n_steps and num_epochs must be set for training with RLDS dataset"

    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps

    # add info to optim_params
    with config.values_unlocked():
        if "optim_params" in config.algo:
            # add info to optim_params of each net
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = len(trainset) if train_num_steps is None else train_num_steps
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device
    )

    if resume:
        # load ckpt dict
        print("*" * 50)
        print("resuming from ckpt at {}".format(latest_model_path))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_path)
        except Exception as e:
            print("got error: {} when loading from {}".format(e, latest_model_path))
            print("trying backup path {}".format(latest_model_backup_path))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_backup_path)
        # load model weights and optimizer state
        model.deserialize(ckpt_dict["model"], load_optimizers=True)
        print("*" * 50)
    
    # if checkpoint is specified, load in model weights;
    # will not use ckpt_path if resuming training
    ckpt_path = config.experiment.ckpt_path
    if (ckpt_path is not None) and (not resume):
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)
    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")
    last_ckpt_time = time.time()

    start_epoch = 1 # epoch numbers start at 1
    if resume:
        # load variable state needed for train loop
        variable_state = ckpt_dict["variable_state"]
        start_epoch = variable_state["epoch"] + 1 # start at next epoch, since this recorded the last epoch of training completed
        print("*" * 50)
        print("resuming training from epoch {}".format(start_epoch))
        print("*" * 50)

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        if should_save_ckpt:
            last_ckpt_time = time.time()

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # get variable state for saving model
        variable_state = dict(
            epoch=epoch,
        )

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:    
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=None,
                shape_meta=shape_meta,
                variable_state=variable_state,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        # always save latest model for resume functionality
        print("\nsaving latest model at {}...\n".format(latest_model_path))
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=None,
            shape_meta=shape_meta,
            variable_state=variable_state,
            ckpt_path=latest_model_path,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
        )

        # keep a backup model in case last.pth is malformed (e.g. job died last time during saving)
        shutil.copyfile(latest_model_path, latest_model_backup_path)
        print("\nsaved backup of latest model at {}\n".format(latest_model_backup_path))

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = [{"path": args.dataset}]

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device, resume=args.resume)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    # resume training from latest checkpoint
    parser.add_argument(
        "--resume",
        action='store_true',
        help="set this flag to resume training from latest checkpoint",
    )

    args = parser.parse_args()
    main(args)