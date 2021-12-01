## Requirements
- This codebase is written for `python3`.
- To install necessary python packages, run `pip install -r requirements.txt`.

### Data
- Please download the data before running the code

### Training SSL models
- Code for training SSL models are in the following file: [`ssl_mocov2_only.py`](./ssl_mocov2_only.py) for MoCo-only model, [`ssl_mocov2_mse.py`](./ssl_mocov2_mse.py) for MoCo + MSE, and [`ssl_mocov2_bt.py`](./ssl_mocov2_bt.py) for MoCo + Barlow Twins
```
usage: ssl_mocov2_mse.py [--data-path] [--batch-size] [--lamb_values] [--save-path]

  arguments:
    --data-path                  path to data directory (required)
    --save-path                  path to saving model directory (required)
    --batch-size                 train batch size (default: 32)
    --lamb_values                lamb for mse loss from each block of resnet50 (default: [0.25,0.25,0.25,0.25]) (not present for MoCo only models)
```

### Finetuning SSL models and saving the intermediate features and predictions
- Code for fine-tuning SSL models is in the following file: [`fine_tune_mocov2.py`](./fine_tune_mocov2.py)
```
usage: fine_tune_mocov2.py [--ckpt_path] [--data-path] [--data_size] [--batch-size] [--save-path] [--file_extension] [--mse_btwin] [--from_scratch] [--only_ll]

  arguments:
    --ckpt_path                  path to the ssl check point model used of intializing wts. of the model (not required for models trained from scratch)
    --data-path                  path to data directory (required)
    --data_size                  fraction of data, 1 means 100% and 0.01 means 1% (default = 1)
    --batch-size                 train batch size (default: 32)
    --save-path                  path to saving model directory (required)
    --file_extension             any file extensions you want to add to the model saving file name (default: None)
    --mse_btwin                  could be one of 'moco-mse', 'moco-btwin' or 'moco-only' (default = None)
    --from_scratch               if true then a fully supervised model with random intialization will be trained (defualt = False)
    --only_ll                    if true, then it will only train the last linear layer (rest of the network would be freezed) (default = False)
```
This will have an output file which would be a dictionary that has intermediate features, predictions, ground truth labels, mean performance (with 95% confidence limits) for the test set. The name of dictionary would end with `_preds.p`.
