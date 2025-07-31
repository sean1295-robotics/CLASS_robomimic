# robomimic fork for CLASS

## Installation
Install using the following commands
```
$ conda create -n CLASS python=3.11
$ conda activate CLASS
$ conda install pytorch==2.6.0 torchvision==0.21.0 -c pytorch
$ git clone https://github.com/sean1295-robotics/CLASS_robomimic.git
$ cd CLASS_robomimic
$ pip install -e .
```
## Prepare Dataset
```
$ python robomimic/scripts/download_datasets.py \ 
--tasks square \ 
--dataset_types ph \ 
--hdf5_types low_dim
$ python robomimic/scripts/dataset_states_to_obs.py \ 
--dataset robomimic/datasets/square/ph/low_dim_v15.hdf5 \ 
--output_name robomimic/datasets/square/ph/image_v15.hdf5 \ 
--camera_names agentview \ 
--camera_height 256 \ 
--camera_width 256
$ python robomimic/scripts/conversion/robosuite_add_absolute_actions.py \ 
--dataset robomimic/datasets/square/ph/image_v15.hdf5
$ python robomimic/scripts/extract_action_dict.py \ 
--dataset robomimic/datasets/square/ph/image_v15.hdf5 
```
-------
## Precompute and Pretrain
```
$ python robomimic/scripts/train.py \
--dataset robomimic/datasets/square/ph/image_v15.hdf5 \
--config robomimic/exps/templates/diffusion_policy_pretrain.json \
--pretrain
```
-------
## Finetune
Change `"experiment/ckpt_path"` argument from `robomimic/exps/templates/diffusion_policy_finetune.json` by selecting the last (or best) checkpoint from pretraining. (e.g., `robomimic/exps/pretrain/dp_imn_pretrain/YYYYMMDDHHMMSS/models/model_epoch_50_image_v15_success_0.92.pth`) and run the following command.
```
$ python robomimic/scripts/train.py \
--dataset robomimic/datasets/square/ph/image_v15.hdf5 \
--config robomimic/exps/templates/diffusion_policy_finetune.json
```

<!-- ## Citation

Please cite [this paper](https://arxiv.org/abs/2108.03298) if you use this framework in your work:

```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
``` -->
