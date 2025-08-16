"""
Contains torch Modules for core observation processing blocks
such as encoders (e.g. EncoderCore, VisualCore, ScanCore, ...)
and randomizers (e.g. Randomizer, CropRandomizer).
"""

import abc
import numpy as np
import textwrap
import random

import torch
import torch.nn as nn

import robomimic.models.base_nets as BaseNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict

# NOTE: this is required for the backbone classes to be found by the `eval` call in the core networks
from robomimic.models.base_nets import *
from robomimic.utils.vis_utils import visualize_image_randomizer
from robomimic.macros import VISUALIZE_RANDOMIZER

import torchvision.transforms.functional as TVF
from torchvision.transforms import Lambda, Compose

"""
================================================
Encoder Core Networks (Abstract class)
================================================
"""
class EncoderCore(BaseNets.Module):
    """
    Abstract class used to categorize all cores used to encode observations
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(EncoderCore, self).__init__()

    def __init_subclass__(cls, **kwargs):
        """
        Hook method to automatically register all valid subclasses so we can keep track of valid observation encoders
        in a global dict.

        This global dict stores mapping from observation encoder network name to class.
        We keep track of these registries to enable automated class inference at runtime, allowing
        users to simply extend our base encoder class and refer to that class in string form
        in their config, without having to manually register their class internally.
        This also future-proofs us for any additional encoder classes we would
        like to add ourselves.
        """
        ObsUtils.register_encoder_core(cls)


"""
================================================
Visual Core Networks (Backbone + Pool)
================================================
"""
class VisualCore(EncoderCore, BaseNets.ConvBase):
    """
    A network block that combines a visual backbone network with optional pooling
    and linear layers.
    """
    def __init__(
        self,
        input_shape,
        backbone_class="ResNet18Conv",
        pool_class="SpatialSoftmax",
        backbone_kwargs=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=64,
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            backbone_class (str): class name for the visual backbone network. Defaults
                to "ResNet18Conv".
            pool_class (str): class name for the visual feature pooler (optional)
                Common options are "SpatialSoftmax" and "SpatialMeanPool". Defaults to
                "SpatialSoftmax".
            backbone_kwargs (dict): kwargs for the visual backbone network (optional)
            pool_kwargs (dict): kwargs for the visual feature pooler (optional)
            flatten (bool): whether to flatten the visual features
            feature_dimension (int): if not None, add a Linear layer to
                project output into a desired feature dimension
        """
        super(VisualCore, self).__init__(input_shape=input_shape)
        self.flatten = flatten

        if backbone_kwargs is None:
            backbone_kwargs = dict()

        # add input channel dimension to visual core inputs
        backbone_kwargs["input_channel"] = input_shape[0]

        # extract only relevant kwargs for this specific backbone
        backbone_kwargs = extract_class_init_kwargs_from_dict(
                cls = ObsUtils.OBS_ENCODER_BACKBONES[backbone_class],
                dic=backbone_kwargs, copy=True)

        # visual backbone
        assert isinstance(backbone_class, str)
        self.backbone = eval(backbone_class)(**backbone_kwargs)

        assert isinstance(self.backbone, BaseNets.ConvBase)

        feat_shape = self.backbone.output_shape(input_shape)
        net_list = [self.backbone]

        # maybe make pool net
        if pool_class is not None:
            assert isinstance(pool_class, str)
            # feed output shape of backbone to pool net
            if pool_kwargs is None:
                pool_kwargs = dict()
            # extract only relevant kwargs for this specific backbone
            pool_kwargs["input_shape"] = feat_shape
            pool_kwargs = extract_class_init_kwargs_from_dict(cls=eval(pool_class), dic=pool_kwargs, copy=True)
            self.pool = eval(pool_class)(**pool_kwargs)
            assert isinstance(self.pool, BaseNets.Module)

            feat_shape = self.pool.output_shape(feat_shape)
            net_list.append(self.pool)
        else:
            self.pool = None

        # flatten layer
        if self.flatten:
            net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # maybe linear layer
        self.feature_dimension = feature_dimension
        if feature_dimension is not None:
            assert self.flatten
            linear = torch.nn.Linear(int(np.prod(feat_shape)), feature_dimension)
            net_list.append(linear)

        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        if self.feature_dimension is not None:
            # linear output
            return [self.feature_dimension]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            # pool output
            feat_shape = self.pool.output_shape(feat_shape)
        # backbone + flat output
        if self.flatten:
            return [np.prod(feat_shape)]
        else:
            return feat_shape

    def forward(self, inputs, lang_emb=None):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        return super(VisualCore, self).forward(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone), indent)
        msg += textwrap.indent("\npool_net={}".format(self.pool), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class VisualCoreLanguageConditioned(VisualCore):
    """
    Variant of VisualCore that expects language embedding during forward pass.
    """
    def __init__(
        self,
        input_shape,
        backbone_class="ResNet18ConvFiLM",
        pool_class="SpatialSoftmax",
        backbone_kwargs=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=64,
    ):
        """
        Update default backbone class.
        """
        super(VisualCoreLanguageConditioned, self).__init__(
            input_shape=input_shape,
            backbone_class=backbone_class,
            pool_class=pool_class,
            backbone_kwargs=backbone_kwargs,
            pool_kwargs=pool_kwargs,
            flatten=flatten,
            feature_dimension=feature_dimension,
        )

    def forward(self, inputs, lang_emb=None):
        """
        Update forward pass to pass language embedding through ResNet18ConvFiLM.
        """
        assert lang_emb is not None
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)

        # feed lang_emb through backbone explicitly, and then feed through rest of network
        assert self.backbone is not None
        x = self.backbone(inputs, lang_emb)
        x = self.nets[1:](x)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x


"""
================================================
Scan Core Networks (Conv1D Sequential + Pool)
================================================
"""
class ScanCore(EncoderCore, BaseNets.ConvBase):
    """
    A network block that combines a Conv1D backbone network with optional pooling
    and linear layers.
    """
    def __init__(
        self,
        input_shape,
        conv_kwargs=None,
        conv_activation="relu",
        pool_class=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=None,
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            conv_kwargs (dict): kwargs for the conv1d backbone network. Should contain lists for the following values:
                out_channels (int)
                kernel_size (int)
                stride (int)
                ...

                If not specified, or an empty dictionary is specified, some default settings will be used.
            conv_activation (str or None): Activation to use between conv layers. Default is relu.
                Currently, valid options are {relu}
            pool_class (str): class name for the visual feature pooler (optional)
                Common options are "SpatialSoftmax" and "SpatialMeanPool"
            pool_kwargs (dict): kwargs for the visual feature pooler (optional)
            flatten (bool): whether to flatten the network output
            feature_dimension (int): if not None, add a Linear layer to
                project output into a desired feature dimension (note: flatten must be set to True!)
        """
        super(ScanCore, self).__init__(input_shape=input_shape)
        self.flatten = flatten
        self.feature_dimension = feature_dimension

        if conv_kwargs is None:
            conv_kwargs = dict()

        # Generate backbone network
        # N input channels is assumed to be the first dimension
        self.backbone = BaseNets.Conv1dBase(
            input_channel=self.input_shape[0],
            activation=conv_activation,
            **conv_kwargs,
        )
        feat_shape = self.backbone.output_shape(input_shape=input_shape)

        # Create netlist of all generated networks
        net_list = [self.backbone]

        # Possibly add pooling network
        if pool_class is not None:
            # Add an unsqueeze network so that the shape is correct to pass to pooling network
            self.unsqueeze = Unsqueeze(dim=-1)
            net_list.append(self.unsqueeze)
            # Get output shape
            feat_shape = self.unsqueeze.output_shape(feat_shape)
            # Create pooling network
            self.pool = eval(pool_class)(input_shape=feat_shape, **pool_kwargs)
            net_list.append(self.pool)
            feat_shape = self.pool.output_shape(feat_shape)
        else:
            self.unsqueeze, self.pool = None, None

        # flatten layer
        if self.flatten:
            net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # maybe linear layer
        if self.feature_dimension is not None:
            assert self.flatten
            linear = torch.nn.Linear(int(np.prod(feat_shape)), self.feature_dimension)
            net_list.append(linear)

        # Generate final network
        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        if self.feature_dimension is not None:
            # linear output
            return [self.feature_dimension]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            # pool output
            feat_shape = self.pool.output_shape(self.unsqueeze.output_shape(feat_shape))
        # backbone + flat output
        return [np.prod(feat_shape)] if self.flatten else feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        return super(ScanCore, self).forward(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone), indent)
        msg += textwrap.indent("\npool_net={}".format(self.pool), indent)
        msg = header + '(' + msg + '\n)'
        return msg


"""
================================================
Observation Randomizer Networks
================================================
"""
class Randomizer(BaseNets.Module):
    """
    Base class for randomizer networks. Each randomizer should implement the @output_shape_in,
    @output_shape_out, @forward_in, and @forward_out methods. The randomizer's @forward_in
    method is invoked on raw inputs, and @forward_out is invoked on processed inputs
    (usually processed by a @VisualCore instance). Note that the self.training property
    can be used to change the randomizer's behavior at train vs. test time.
    """
    def __init__(self):
        super(Randomizer, self).__init__()

    def __init_subclass__(cls, **kwargs):
        """
        Hook method to automatically register all valid subclasses so we can keep track of valid observation randomizers
        in a global dict.

        This global dict stores mapping from observation randomizer network name to class.
        We keep track of these registries to enable automated class inference at runtime, allowing
        users to simply extend our base randomizer class and refer to that class in string form
        in their config, without having to manually register their class internally.
        This also future-proofs us for any additional randomizer classes we would
        like to add ourselves.
        """
        ObsUtils.register_randomizer(cls)

    def output_shape(self, input_shape=None):
        """
        This function is unused. See @output_shape_in and @output_shape_out.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    def forward_in(self, inputs):
        """
        Randomize raw inputs if training.
        """
        if self.training:
            randomized_inputs = self._forward_in(inputs=inputs)
            if VISUALIZE_RANDOMIZER:
                num_samples_to_visualize = min(4, inputs.shape[0])
                self._visualize(inputs, randomized_inputs, num_samples_to_visualize=num_samples_to_visualize)
            return randomized_inputs
        else:
            return self._forward_in_eval(inputs)

    def forward_out(self, inputs):
        """
        Processing for network outputs.
        """
        if self.training:
            return self._forward_out(inputs)
        else:
            return self._forward_out_eval(inputs)

    @abc.abstractmethod
    def _forward_in(self, inputs):
        """
        Randomize raw inputs.
        """
        raise NotImplementedError

    def _forward_in_eval(self, inputs):
        """
        Test-time behavior for the randomizer
        """
        return inputs

    @abc.abstractmethod
    def _forward_out(self, inputs):
        """
        Processing for network outputs.
        """
        return inputs

    def _forward_out_eval(self, inputs):
        """
        Test-time behavior for the randomizer
        """
        return inputs

    @abc.abstractmethod
    def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
        """
        Visualize the original input and the randomized input for _forward_in for debugging purposes.
        """
        pass


class CropRandomizer(Randomizer):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """
    def __init__(
        self,
        input_shape,
        crop_height=76,
        crop_width=76,
        num_crops=1,
        pos_enc=False,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to the output to encode the spatial
                location of the cropped pixels in the source image
        """
        super(CropRandomizer, self).__init__()

        assert len(input_shape) == 3 # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # outputs are shape (C, CH, CW), or maybe C + 2 if using position encoding, because
        # the number of crops are reshaped into the batch dimension, increasing the batch
        # size from B to B * N
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def _forward_in(self, inputs):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
        out, _ = ObsUtils.sample_random_image_crops(
            images=inputs,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        # [B, N, ...] -> [B * N, ...]
        return TensorUtils.join_dimensions(out, 0, 1)

    def _forward_in_eval(self, inputs):
        """
        Do center crops during eval
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
        inputs = inputs.permute(*range(inputs.dim()-3), inputs.dim()-2, inputs.dim()-1, inputs.dim()-3)
        out = ObsUtils.center_crop(inputs, self.crop_height, self.crop_width)
        out = out.permute(*range(out.dim()-3), out.dim()-1, out.dim()-3, out.dim()-2)
        return out

    def _forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_crops)
        out = TensorUtils.reshape_dimensions(inputs, begin_axis=0, end_axis=0,
                                             target_dims=(batch_size, self.num_crops))
        return out.mean(dim=1)

    def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0, batch_size, size=(num_samples_to_visualize,))
        pre_random_input_np = TensorUtils.to_numpy(pre_random_input)[random_sample_inds]
        randomized_input = TensorUtils.reshape_dimensions(
            randomized_input,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size, self.num_crops)
        )  # [B * N, ...] -> [B, N, ...]
        randomized_input_np = TensorUtils.to_numpy(randomized_input[random_sample_inds])

        pre_random_input_np = pre_random_input_np.transpose((0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        randomized_input_np = randomized_input_np.transpose((0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]

        visualize_image_randomizer(
            pre_random_input_np,
            randomized_input_np,
            randomizer_name='{}'.format(str(self.__class__.__name__))
        )

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}], num_crops={})".format(
            self.input_shape, self.crop_height, self.crop_width, self.num_crops)
        return msg
    
import torch.nn.functional as F
from typing import List, Tuple, Optional

class SquareCropNormalizer(Randomizer):
    """
    Crop square regions based on input image height, then resize to 224x224.
    During training: random crop with size (height * 0.9, height * 0.9)
    During eval: high-contrast-focused crop with size (height * 0.9, height * 0.9)
    Assumes input images are rectangular with w > h.
    """
    
    def __init__(
        self,
        crop_factor: float = 0.9,
        output_size: int = 224,
        pos_enc: bool = False,
    ):
        """
        Args:
            crop_factor (float): factor to multiply height by for crop size (default: 0.9)
            output_size (int): final output size after resize (default: 224)
            pos_enc (bool): if True, add 2 channels to encode spatial location
        """
        super(SquareCropNormalizer, self).__init__()
        
        self.crop_factor = crop_factor
        self.output_size = output_size
        self.pos_enc = pos_enc

    def _get_crop_size(self, input_tensor: torch.Tensor) -> int:
        """
        Dynamically calculate crop size based on input height.
        """
        height = input_tensor.shape[-2]        
        crop_size = int(height * self.crop_factor)
        return crop_size

    def output_shape_in(self, input_shape: Optional[Tuple[int, ...]] = None) -> List[int]:
        """
        Compute output shape for forward_in operation.
        """
        if input_shape is not None:
            channels = input_shape[0] if len(input_shape) >= 3 else 3
        else:
            channels = 3  # Default assumption
            
        out_c = channels + 2 if self.pos_enc else channels
        return [out_c, self.output_size, self.output_size]

    def output_shape_out(self, input_shape: Optional[Tuple[int, ...]] = None) -> List[int]:
        """
        Compute output shape for forward_out operation.
        """
        if input_shape is not None:
            return list(input_shape)
        else:
            channels = 3 + 2 if self.pos_enc else 3
            return [channels, self.output_size, self.output_size]

    def _forward_in(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass: random square crop + resize to 224x224.
        """
        assert len(inputs.shape) >= 3, "Input must have at least (C, H, W) dimensions"
        
        # Ensure 4D input for batch processing
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, channels, height, width = inputs.shape
        crop_size = self._get_crop_size(inputs)
        
        # Ensure crop size doesn't exceed image dimensions
        if crop_size > height or crop_size > width:
            raise ValueError(f"Crop size {crop_size} exceeds image dimensions {height}x{width}")
        
        # Random crop coordinates
        max_h_start = height - crop_size
        max_w_start = width - crop_size
        
        # Generate random crops for each image in batch
        crops = []
        for b in range(batch_size):
            h_start = torch.randint(0, max_h_start + 1, (1,)).item() if max_h_start > 0 else 0
            w_start = torch.randint(0, max_w_start + 1, (1,)).item() if max_w_start > 0 else 0
            
            crop = inputs[b:b+1, :, h_start:h_start+crop_size, w_start:w_start+crop_size]
            
            # Add position encoding if requested
            if self.pos_enc:
                crop = self._add_position_encoding(crop, h_start, w_start, height, width)
            
            crops.append(crop)
        
        # Concatenate all crops
        output = torch.cat(crops, dim=0)
        
        # Resize to target size
        output = F.interpolate(output, size=(self.output_size, self.output_size), 
                             mode='bilinear', align_corners=False)
        
        # Handle original 3D input
        if squeeze_output:
            output = output.squeeze(0)
            
        return output

    def _forward_in_eval(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluation forward pass: high-contrast-focused square crop + resize to 224x224.
        Centers the crop on the region with the highest average differential between neighbor pixels.
        """
        assert len(inputs.shape) >= 3, "Input must have at least (C, H, W) dimensions"
        
        # Ensure 4D input for batch processing
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, channels, height, width = inputs.shape
        crop_size = self._get_crop_size(inputs)
        
        # Ensure crop size doesn't exceed image dimensions
        if crop_size > height or crop_size > width:
            raise ValueError(f"Crop size {crop_size} exceeds image dimensions {height}x{width}")
        
        # Process each image in the batch
        crops = []
        for b in range(batch_size):
            h_start, w_start = self._find_high_contrast_crop_center(inputs[b], crop_size)
            
            crop = inputs[b:b+1, :, h_start:h_start+crop_size, w_start:w_start+crop_size]
            
            # Add position encoding if requested
            if self.pos_enc:
                crop = self._add_position_encoding(crop, h_start, w_start, height, width)
            
            crops.append(crop)
        
        # Concatenate all crops
        output = torch.cat(crops, dim=0)
        
        # Resize to target size
        output = F.interpolate(output, size=(self.output_size, self.output_size), 
                             mode='bilinear', align_corners=False)
        
        # Handle original 3D input
        if squeeze_output:
            output = output.squeeze(0)
        
        return output

    def _forward_out(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Pass-through since we only have 1 crop.
        """
        return inputs

    def _find_high_contrast_crop_center(self, image: torch.Tensor, crop_size: int) -> Tuple[int, int]:
        """
        Find the optimal crop position that contains the highest average differential between neighbor pixels.
        
        Args:
            image: Single image tensor of shape (C, H, W)
            crop_size: Size of the square crop
            
        Returns:
            Tuple of (h_start, w_start) coordinates for the crop
        """
        channels, height, width = image.shape
        
        # If crop is larger than or equal to image, return centered crop
        if crop_size >= height or crop_size >= width:
            h_start = max(0, (height - crop_size) // 2)
            w_start = max(0, (width - crop_size) // 2)
            return h_start, w_start

        # Convert to grayscale for contrast calculation
        if channels == 1:
            grayscale = image[0:1, :, :]
        else:
            # Simple luminance conversion: 0.299 * R + 0.587 * G + 0.114 * B
            grayscale = (0.299 * image[0:1, :, :] + 0.587 * image[1:2, :, :] + 0.114 * image[2:3, :, :])
        
        # Add batch dimension for conv2d
        grayscale_4d = grayscale.unsqueeze(0)
        
        # Sobel kernels for horizontal and vertical edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=grayscale_4d.dtype, device=grayscale_4d.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=grayscale_4d.dtype, device=grayscale_4d.device).unsqueeze(0).unsqueeze(0)

        # Compute gradients using convolution
        grad_x = F.conv2d(grayscale_4d, sobel_x, padding=1)
        grad_y = F.conv2d(grayscale_4d, sobel_y, padding=1)

        # Compute the magnitude of the gradient (approximation of contrast)
        contrast_map = torch.sqrt(grad_x**2 + grad_y**2)
        contrast_map = contrast_map.squeeze() # Remove batch and channel dims

        # Use convolution with a uniform kernel to find the sum of contrast within each crop window
        kernel = torch.ones(1, 1, crop_size, crop_size, device=image.device)
        contrast_map_4d = contrast_map.unsqueeze(0).unsqueeze(0)
        
        # Convolve to get the total contrast for each possible crop window
        contrast_sums = F.conv2d(contrast_map_4d, kernel, padding=0)
        contrast_sums = contrast_sums.squeeze()
        
        # Find the position with the maximum sum
        max_idx = torch.argmax(contrast_sums.flatten())
        
        # Convert flat index back to 2D coordinates
        sum_height, sum_width = contrast_sums.shape
        best_h_start = max_idx // sum_width
        best_w_start = max_idx % sum_width
        
        return best_h_start.item(), best_w_start.item()

    def _visualize(self, pre_random_input: torch.Tensor, randomized_input: torch.Tensor, 
                  num_samples_to_visualize: int = 2):
        """
        Visualize the cropping and resizing process.
        """
        # This part of the code is unchanged and assumes a separate visualization function
        # like `visualize_image_randomizer` exists.
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0, batch_size, size=(min(num_samples_to_visualize, batch_size),))
        
        pre_random_input_np = pre_random_input[random_sample_inds].cpu().numpy()
        pre_random_input_np = pre_random_input_np.transpose((0, 2, 3, 1))
        
        randomized_input_np = randomized_input[random_sample_inds].cpu().numpy()
        randomized_input_np = randomized_input_np.transpose((0, 2, 3, 1))
        randomized_input_np = randomized_input_np[:, None, ...]
        
        # NOTE: This assumes the existence of the visualize_image_randomizer function
        # visualize_image_randomizer(
        #     pre_random_input_np,
        #     randomized_input_np,
        #     randomizer_name=f'{self.__class__.__name__}'
        # )

    def __repr__(self) -> str:
        """Pretty print network."""
        header = f'{self.__class__.__name__}'
        msg = (f"{header}(crop_factor={self.crop_factor}, output_size={self.output_size}, "
               f"pos_enc={self.pos_enc})")
        return msg

class DroidRandomizer(Randomizer):
    """
    Randomizer for Droid, which does no-op during training, and performs a center
    crop on inputs during evaluation.
    """
    def __init__(
        self,
        input_shape,
        crop_height=224,
        crop_width=224,
        num_crops=1,
        pos_enc=False,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): ignored. kept for compatibility.
            pos_enc (bool): ignored. kept for compatibility.
        """
        super(DroidRandomizer, self).__init__()

        assert len(input_shape) == 3 # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]
        
        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        out_c = self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return list(input_shape)

    def _forward_in(self, inputs):
        """
        No-op during training.
        """
        return inputs

    def _forward_in_eval(self, inputs):
        """
        Do center crops during eval.
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
        inputs = inputs.permute(*range(inputs.dim()-3), inputs.dim()-2, inputs.dim()-1, inputs.dim()-3)
        out = ObsUtils.center_crop(inputs, self.crop_height, self.crop_width)
        out = out.permute(*range(out.dim()-3), out.dim()-1, out.dim()-3, out.dim()-2)
        return out

    def _forward_out(self, inputs):
        """
        No-op during training.
        """
        return inputs

    def _forward_out_eval(self, inputs):
        """
        No-op during evaluation.
        """
        return inputs
    
    def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
        """
        No-op during training, as per the spec.
        """
        pass

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}])".format(
            self.input_shape, self.crop_height, self.crop_width)
        return msg
    
class ColorRandomizer(Randomizer):
    """
    Randomly sample color jitter at input, and then average across color jtters at output.
    """
    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        num_samples=1,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            brightness (None or float or 2-tuple): How much to jitter brightness. brightness_factor is chosen uniformly
                from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.
            contrast (None or float or 2-tuple): How much to jitter contrast. contrast_factor is chosen uniformly
                from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.
            saturation (None or float or 2-tuple): How much to jitter saturation. saturation_factor is chosen uniformly
                from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.
            hue (None or float or 2-tuple): How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or
                the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. To jitter hue, the pixel
                values of the input image has to be non-negative for conversion to HSV space; thus it does not work
                if you normalize your image to an interval with negative values, or use an interpolation that
                generates negative values before using this function.
            num_samples (int): number of random color jitters to take
        """
        super(ColorRandomizer, self).__init__()

        assert len(input_shape) == 3 # (C, H, W)

        self.input_shape = input_shape
        self.brightness = [max(0, 1 - brightness), 1 + brightness] if type(brightness) in {float, int} else brightness
        self.contrast = [max(0, 1 - contrast), 1 + contrast] if type(contrast) in {float, int} else contrast
        self.saturation = [max(0, 1 - saturation), 1 + saturation] if type(saturation) in {float, int} else saturation
        self.hue = [-hue, hue] if type(hue) in {float, int} else hue
        self.num_samples = num_samples

    @torch.jit.unused
    def get_transform(self):
        """
        Get a randomized transform to be applied on image.

        Implementation taken directly from:

        https://github.com/pytorch/vision/blob/2f40a483d73018ae6e1488a484c5927f2b309969/torchvision/transforms/transforms.py#L1053-L1085

        Returns:
            Transform: Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(Lambda(lambda img: TVF.adjust_brightness(img, brightness_factor)))

        if self.contrast is not None:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(Lambda(lambda img: TVF.adjust_contrast(img, contrast_factor)))

        if self.saturation is not None:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(Lambda(lambda img: TVF.adjust_saturation(img, saturation_factor)))

        if self.hue is not None:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(Lambda(lambda img: TVF.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def get_batch_transform(self, N):
        """
        Generates a batch transform, where each set of sample(s) along the batch (first) dimension will have the same
        @N unique ColorJitter transforms applied.

        Args:
            N (int): Number of ColorJitter transforms to apply per set of sample(s) along the batch (first) dimension

        Returns:
            Lambda: Aggregated transform which will autoamtically apply a different ColorJitter transforms to
                each sub-set of samples along batch dimension, assumed to be the FIRST dimension in the inputted tensor
                Note: This function will MULTIPLY the first dimension by N
        """
        return Lambda(lambda x: torch.stack([self.get_transform()(x_) for x_ in x for _ in range(N)]))

    def output_shape_in(self, input_shape=None):
        # outputs are same shape as inputs
        return list(input_shape)

    def output_shape_out(self, input_shape=None):
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def _forward_in(self, inputs):
        """
        Samples N random color jitters for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions

        # Make sure shape is exactly 4
        if len(inputs.shape) == 3:
            inputs = torch.unsqueeze(inputs, dim=0)

        # Create lambda to aggregate all color randomizings at once
        transform = self.get_batch_transform(N=self.num_samples)

        return transform(inputs)

    def _forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_samples)
        out = TensorUtils.reshape_dimensions(inputs, begin_axis=0, end_axis=0,
                                             target_dims=(batch_size, self.num_samples))
        return out.mean(dim=1)

    def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0, batch_size, size=(num_samples_to_visualize,))
        pre_random_input_np = TensorUtils.to_numpy(pre_random_input)[random_sample_inds]
        randomized_input = TensorUtils.reshape_dimensions(
            randomized_input,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size, self.num_samples)
        )  # [B * N, ...] -> [B, N, ...]
        randomized_input_np = TensorUtils.to_numpy(randomized_input[random_sample_inds])

        pre_random_input_np = pre_random_input_np.transpose((0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        randomized_input_np = randomized_input_np.transpose((0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]

        visualize_image_randomizer(
            pre_random_input_np,
            randomized_input_np,
            randomizer_name='{}'.format(str(self.__class__.__name__))
        )

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + f"(input_shape={self.input_shape}, brightness={self.brightness}, contrast={self.contrast}, " \
                       f"saturation={self.saturation}, hue={self.hue}, num_samples={self.num_samples})"
        return msg


class GaussianNoiseRandomizer(Randomizer):
    """
    Randomly sample gaussian noise at input, and then average across noises at output.
    """
    def __init__(
        self,
        input_shape,
        noise_mean=0.0,
        noise_std=0.3,
        limits=(0,255),
        num_samples=1,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            noise_mean (float): Mean of noise to apply
            noise_std (float): Standard deviation of noise to apply
            limits (None or 2-tuple): If specified, should be the (min, max) values to clamp all noisied samples to
            num_samples (int): number of random color jitters to take
        """
        super(GaussianNoiseRandomizer, self).__init__()

        self.input_shape = input_shape
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.limits = limits
        self.num_samples = num_samples

    def output_shape_in(self, input_shape=None):
        # outputs are same shape as inputs
        return list(input_shape)

    def output_shape_out(self, input_shape=None):
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def _forward_in(self, inputs):
        """
        Samples N random gaussian noises for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        out = TensorUtils.repeat_by_expand_at(inputs, repeats=self.num_samples, dim=0)

        # Sample noise across all samples
        out = torch.rand(size=out.shape).to(inputs.device) * self.noise_std + self.noise_mean + out

        # Possibly clamp
        if self.limits is not None:
            out = torch.clip(out, min=self.limits[0], max=self.limits[1])

        return out

    def _forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_samples)
        out = TensorUtils.reshape_dimensions(inputs, begin_axis=0, end_axis=0,
                                             target_dims=(batch_size, self.num_samples))
        return out.mean(dim=1)

    def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0, batch_size, size=(num_samples_to_visualize,))
        pre_random_input_np = TensorUtils.to_numpy(pre_random_input)[random_sample_inds]
        randomized_input = TensorUtils.reshape_dimensions(
            randomized_input,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size, self.num_samples)
        )  # [B * N, ...] -> [B, N, ...]
        randomized_input_np = TensorUtils.to_numpy(randomized_input[random_sample_inds])

        pre_random_input_np = pre_random_input_np.transpose((0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        randomized_input_np = randomized_input_np.transpose((0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]

        visualize_image_randomizer(
            pre_random_input_np,
            randomized_input_np,
            randomizer_name='{}'.format(str(self.__class__.__name__))
        )

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + f"(input_shape={self.input_shape}, noise_mean={self.noise_mean}, noise_std={self.noise_std}, " \
                       f"limits={self.limits}, num_samples={self.num_samples})"
        return msg

class CompoundRandomizer(Randomizer):
    """
    Combines CropRandomizer, ColorRandomizer, and GaussianNoiseRandomizer into a single randomizer.
    Applies all three augmentations sequentially at input, and averages across all samples at output.
    """
    def __init__(
        self,
        input_shape,
        # Crop randomizer params
        crop_height=224,
        crop_width=224,
        num_crops=1,
        pos_enc=False,
        # Color randomizer params 
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        num_color_samples=1,
        # Gaussian noise randomizer params 
        noise_mean=0.0,
        noise_std=5,
        noise_limits=(0,255),
        num_noise_samples=1,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height for CropRandomizer
            crop_width (int): crop width for CropRandomizer
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to encode spatial location
            brightness (float): brightness jitter for ColorRandomizer
            contrast (float): contrast jitter for ColorRandomizer
            saturation (float): saturation jitter for ColorRandomizer
            hue (float): hue jitter for ColorRandomizer
            num_color_samples (int): number of color jitter samples
            noise_mean (float): mean of gaussian noise
            noise_std (float): standard deviation of gaussian noise
            noise_limits (None or 2-tuple): min/max values to clamp noisy samples
            num_noise_samples (int): number of noise samples
        """
        super(CompoundRandomizer, self).__init__()
        
        self.input_shape = input_shape
        
        # Create individual randomizers
        self.crop_randomizer = CropRandomizer(
            input_shape=input_shape,
            crop_height=crop_height,
            crop_width=crop_width,
            num_crops=num_crops,
            pos_enc=pos_enc,
        )
        
        # Get the output shape from crop randomizer for next randomizer
        crop_output_shape = self.crop_randomizer.output_shape_in(input_shape)
        
        self.color_randomizer = ColorRandomizer(
            input_shape=crop_output_shape,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            num_samples=num_color_samples,
        )
        
        self.noise_randomizer = GaussianNoiseRandomizer(
            input_shape=crop_output_shape,
            noise_mean=noise_mean,
            noise_std=noise_std,
            limits=noise_limits,
            num_samples=num_noise_samples,
        )
        
        # Calculate total number of samples (multiplicative effect)
        self.total_samples = num_crops * num_color_samples * num_noise_samples
        
    def output_shape_in(self, input_shape=None):
        """
        Output shape after forward_in operation.
        """
        return self.crop_randomizer.output_shape_in(input_shape)
    
    def output_shape_out(self, input_shape=None):
        """
        Output shape after forward_out operation.
        """
        return list(input_shape)
    
    def _forward_in(self, inputs):
        """
        Apply all three randomizers sequentially.
        """
        # Apply crop randomizer first
        out = self.crop_randomizer._forward_in(inputs)
        
        # Apply color randomizer
        out = self.color_randomizer._forward_in(out)
        
        # Apply gaussian noise randomizer
        out = self.noise_randomizer._forward_in(out)
        
        return out
    
    def _forward_in_eval(self, inputs):
        """
        During evaluation, only apply center crop (no color jitter or noise).
        """
        return self.crop_randomizer._forward_in_eval(inputs)
    
    def _forward_out(self, inputs):
        """
        Average across all samples to get consistent output shape.
        """
        # Calculate original batch size
        batch_size = inputs.shape[0] // self.total_samples
        
        # Reshape to [B, total_samples, ...] and average
        out = TensorUtils.reshape_dimensions(
            inputs, 
            begin_axis=0, 
            end_axis=0,
            target_dims=(batch_size, self.total_samples)
        )
        return out.mean(dim=1)
    
    def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
        """
        Visualize the compound randomization effect.
        """
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0, batch_size, size=(num_samples_to_visualize,))
        pre_random_input_np = TensorUtils.to_numpy(pre_random_input)[random_sample_inds]
        
        # Reshape randomized input to show all samples
        randomized_input = TensorUtils.reshape_dimensions(
            randomized_input,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size, self.total_samples)
        )  # [B * total_samples, ...] -> [B, total_samples, ...]
        randomized_input_np = TensorUtils.to_numpy(randomized_input[random_sample_inds])
        
        # Transpose for visualization
        pre_random_input_np = pre_random_input_np.transpose((0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        randomized_input_np = randomized_input_np.transpose((0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]
        
        visualize_image_randomizer(
            pre_random_input_np,
            randomized_input_np,
            randomizer_name='{}'.format(str(self.__class__.__name__))
        )
    
    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + f"(input_shape={self.input_shape}, total_samples={self.total_samples})\n"
        msg += f"  - CropRandomizer: {self.crop_randomizer}\n"
        msg += f"  - ColorRandomizer: {self.color_randomizer}\n"
        msg += f"  - GaussianNoiseRandomizer: {self.noise_randomizer}"
        return msg