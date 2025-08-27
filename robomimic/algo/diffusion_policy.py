"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
from copy import deepcopy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            feature_activation=None,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
            self.prediction_type = self.algo_config.ddpm.prediction_type
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
            self.prediction_type = self.algo_config.ddim.prediction_type
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        ema_nets = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(parameters=nets.parameters(), power=self.algo_config.ema.power)
            ema_nets = deepcopy(nets)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.ema_nets = ema_nets
        self.ema_nets.eval()
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        
        if self.algo_config.class_weight:
            distance = torch.load('/scratch/dcs3zc/droid_101/joint_distance_l2_32.pth', weights_only=False)
            print('/scratch/dcs3zc/droid_101/joint_distance_l2_32.pth', distance.shape)
            # Initialize normalizer
            from robomimic.utils.class_utils import PairwiseDistanceCDFNormalizer
            self.dist_normalizer = PairwiseDistanceCDFNormalizer(
                torch.from_numpy(distance), 
                quantile=self.algo_config.class_quantile,
                n_quantiles=1000
            ).to_device('cuda')
            del distance
            import gc
            gc.collect()
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        if "lang_emb" in batch["obs"] and isinstance(batch["obs"]["lang_emb"][0], bytes):
            lang_emb_decoded = [inst.decode() if len(inst) else "Do something useful" for inst in batch["obs"]["lang_emb"]]                
            batch["obs"]["lang_emb"] = LangUtils.batch_get_lang_emb(lang_emb_decoded)
            batch["obs"]["lang_emb"] = batch["obs"]["lang_emb"].unsqueeze(1).repeat(1, To, 1)
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def postprocess_batch_for_training(self, batch, obs_normalization_stats):
        """
        Does some operations (like channel swap, uint8 to float conversion, normalization)
        after @process_batch_for_training is called, in order to ensure these operations
        take place on GPU.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader. Assumed to be on the device where
                training will occur (after @process_batch_for_training
                is called)

            obs_normalization_stats (dict or None): if provided, this should map observation 
                keys to dicts with a "mean" and "std" of shape (1, ...) where ... is the 
                default shape for the observation.

        Returns:
            batch (dict): postproceesed batch
        """

        # ensure obs_normalization_stats are torch Tensors on proper device
        obs_normalization_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(obs_normalization_stats), self.device))

        # we will search the nested batch dictionary for the following special batch dict keys
        # and apply the processing function to their values (which correspond to observations)
        obs_keys = ["obs", "next_obs", "goal_obs"]

        def recurse_helper(d):
            """
            Apply process_obs_dict to values in nested dictionary d that match a key in obs_keys.
            """
            for k in d:
                if k in obs_keys:
                    # found key - stop search and process observation
                    if d[k] is not None:
                        d[k] = ObsUtils.process_obs_dict(d[k])
                        if obs_normalization_stats is not None:
                            d[k] = ObsUtils.normalize_dict(d[k], normalization_stats=obs_normalization_stats)
                elif isinstance(d[k], dict):
                    # search down into dictionary
                    recurse_helper(d[k])

        recurse_helper(batch)
        return batch
    
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)      
            ### CLASS LOSS            
            B = batch["actions"].shape[0]                     
            actions = batch['actions']
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]            
            
            obs_cond = obs_features.flatten(start_dim=1)
            
            if self.algo_config.class_weight:
                processed_actions = actions.clone().detach()
                # processed_actions[..., 3:9] /= 2
                
                # Option : l2
                processed_actions_flat = processed_actions.flatten(start_dim = 1)  # Shape: (B, T*D)
                pairwise_distance = torch.cdist(processed_actions_flat, processed_actions_flat, p=2)  # Shape: (B, B)
                
                self_mask = torch.eye(pairwise_distance.shape[0]).to(pairwise_distance.device).bool()
                pairwise_distance = pairwise_distance.masked_fill(self_mask, float('inf')).to(self.device)

                cdf_vals = self.dist_normalizer.distance_to_cdf(pairwise_distance)
                soft_weights = 1-cdf_vals
                pos_mask = soft_weights > 0
                
                # only use image features
                image_dim = sum([v.output_shape(v.input_shape)[0] for k,v in self.nets["policy"]["obs_encoder"].nets['obs'].obs_nets.items() if 'cam' in k])
                obs_cond_normalized = F.normalize(obs_cond[:, :image_dim], dim=1) 
                
                temperature = 0.07
                sim_matrix = torch.div(torch.matmul(obs_cond_normalized, obs_cond_normalized.T), temperature)
                
                self_mask = torch.eye(B, device=self.device, dtype=torch.bool)
                pos_mask = pos_mask & ~self_mask
                
                # Compute log-softmax as before
                with torch.no_grad():
                    logits_max, _ = torch.max(sim_matrix.masked_fill(self_mask, -float('inf')), dim=1, keepdim=True)
                
                scaled_sim_stable = sim_matrix - logits_max
                log_denom = torch.logsumexp(scaled_sim_stable.masked_fill(self_mask, float('-inf')), dim=1, keepdim=True)
                log_prob = scaled_sim_stable - log_denom
                
                pos_weights = soft_weights.to(self.device) * pos_mask
                pos_denom = pos_weights.sum(dim=1)
                valid_samples_mask = pos_denom > 1e-6
                
                if not valid_samples_mask.any():
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
                
                numerator = (pos_weights[valid_samples_mask] * log_prob[valid_samples_mask]).sum(dim=1)
                class_loss = - (numerator / pos_denom[valid_samples_mask]).mean()    
            else:               
                class_loss = torch.tensor([0]).to(actions.device)  
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
                        
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            prediction = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            if self.prediction_type == "epsilon":
                bc_loss = F.mse_loss(prediction, noise)
            elif self.prediction_type == "sample":
                bc_loss = F.mse_loss(prediction, actions)
            elif self.prediction_type == "v_prediction":
                velocity = self.noise_scheduler.get_velocity(actions, noise, timesteps)
                bc_loss= F.mse_loss(prediction, velocity)
            else:
                raise TypeError(f"Prediction type: {self.prediction_type} not recognized.")
                     
            loss = bc_loss + class_loss * self.algo_config.class_weight
            
            # logging
            losses = {
                "bc_loss": bc_loss,
                "class_loss": class_loss,
                "loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets.parameters())
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms,
                    "average_soft_weight": soft_weights.mean().cpu() if self.algo_config.class_weight else 0.0
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["loss"].item()
        log["BC_Loss"] = info["losses"]["bc_loss"].item()
        log["CLASS_Loss"] = info["losses"]["class_loss"].item()            
        log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        log["average_soft_weight"] = info["average_soft_weight"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        # TODO: obs_queue already handled by frame_stack
        # make sure we have at least To observations in obs_queue
        # if not enough, repeat
        # if already full, append one to the obs_queue
        # n_repeats = max(To - len(self.obs_queue), 1)
        # self.obs_queue.extend([obs_dict] * n_repeats)
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # turn obs_queue into dict of tensors (concat at T dim)
            # import pdb; pdb.set_trace()
            # obs_dict_list = TensorUtils.list_of_flat_dict_to_dict_of_list(list(self.obs_queue))
            # obs_dict_tensor = dict((k, torch.cat(v, dim=0).unsqueeze(0)) for k,v in obs_dict_list.items())
            
            # run inference
            # [1,T,Da]
            start = To - 1
            end = start + Ta
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)[0,start:end]
            # put actions into the queue
            self.action_queue.extend(action_sequence)
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None, validate=False):
        if not validate:
            assert not self.nets.training
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema_nets
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        return naction

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        if self.ema is not None:
            self.ema.copy_to(self.ema_nets.parameters())
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.load_state_dict(model_dict["ema"])
            self.ema_nets.load_state_dict(model_dict["nets"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
