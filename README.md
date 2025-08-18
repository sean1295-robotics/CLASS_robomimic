# robomimic fork for CLASS

## Installation
Install using the following commands
```
$ bash ./scripts/setup.sh
```
## Prepare Dataset
```
$ bash ./scripts/prepare_dataset.sh
```
-------
## Precompute and Pretrain
```
$ bash ./scripts/precompute_pretrain.sh
```
-------
## Finetune
Change `"experiment/ckpt_path"` argument from `robomimic/exps/templates/diffusion_policy_finetune.json` by selecting the last (or best) checkpoint from pretraining. (e.g., `logs/dp_imn_pretrain/YYYYMMDDHHMMSS/models/model_epoch_50_image_v15_success_0.92.pth`) and run the following command.
```
$ bash ./scripts/finetune.sh
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
