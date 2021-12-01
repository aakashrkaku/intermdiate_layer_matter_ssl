## Requirements
- This codebase is written for `python3`.
- To install necessary python packages, run `pip install -r requirements.txt`.

### Data
- Please download the data before running the code

### Training
- Code for training SSL models are in the following file: [`ssl_mocov2_only.py`](./ssl_mocov2_only.py) for MoCo-only model, [`ssl_mocov2_mse.py`](./ssl_mocov2_mse.py) for MoCo + MSE, and [`ssl_mocov2_bt.py`](./ssl_mocov2_bt.py) for MoCo + Barlow Twins
```
usage: train.py [-c] [-r] [-d] [--lr learning_rate] [--bs batch_size] [--beta beta] [--lambda lambda] [--malpha mixup_alpha]
                [--percent percent] [--asym asym] [--ealpha ema_alpha]  [--name exp_name] 

  arguments:
    -c, --config                  config file path (default: None)
    -r, --resume                  path to latest checkpoint (default: None)
    -d, --device                  indices of GPUs to enable (default: all)     
  
  options:
    --lr learning_rate            learning rate (default value is the value in the config file)
    --bs batch_size               batch size (default value is the value in the config file)
    --beta beta                   temporal ensembling momentum beta for target estimation
    --lambda lambda               regularization coefficient
    --malpha mixup_alpha          mixup parameter alpha
    --percent percent             noise level (e.g. 0.4 for 40%)
    --asym asym                   asymmetric noise is used when set to True
    --ealpha ema_alpha            weight averaging momentum for target estimation
     --name exp_name              experiment name
```
Configuration file is **required** to be specified. Default option values, if not reset, will be the values in the configuration file. 
Examples for ELR and ELR+ are shown in the *readme.md* of `ELR` and `ELR_plus` subfolders respectively.
