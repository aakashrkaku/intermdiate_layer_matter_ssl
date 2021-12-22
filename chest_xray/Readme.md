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
usage: fine_tune_mocov2.py [--ckpt-path] [--data-path] [--data-size] [--batch-size] [--save-path] [--file-extension] [--mse-btwin] [--from-scratch] [--only-ll] [--no-training] [--max-epochs]

  arguments:
    --ckpt-path                  path to the ssl check point model used of intializing wts. of the model (not required for models trained from scratch)
    --data-path                  path to data directory (required)
    --data-size                  fraction of data, 1 means 100% and 0.01 means 1% (default = 1)
    --batch-size                 train batch size (default: 32)
    --save-path                  path to saving model directory (required)
    --file-extension             any file extensions you want to add to the model saving file name (default: None)
    --mse-btwin                  could be one of 'moco-mse', 'moco-btwin' or 'moco-only' (default = None)
    --from-scratch               if true then a fully supervised model with random intialization will be trained (defualt = False)
    --only-ll                    if true, then it will only train the last linear layer (rest of the network would be freezed) (default = False)
    --no-training                if true, then models would not be fine-tuned (default = False)
    --max-epochs                 Max epochs (default = 100)
```
This will have an output file which would be a dictionary that has intermediate features, predictions, ground truth labels, mean performance (with 95% confidence limits) for the test set. The name of dictionary would end with `_preds.p`.

The saved intermediate features can be compared to the features before fine-tuning using the following code. To obtain features before fine tuning, run the above `fine_tune_mocov2.py` with `--no_training` as `True`.

```
from CKA import kernel_CKA, linear_CKA
def bootstrap_similarity(feat1,feat2,num_samples_per_set=1000,num_sets = 50):
    all_sim = []
    size = feat1.shape[0]
    idx = np.arange(size) 
    for i in range(num_sets):
        np.random.shuffle(idx)
        select_idx = idx[:num_samples_per_set]
        all_sim.append(kernel_CKA(feat1[select_idx],feat2[select_idx],sigma=0.8))
    mean_sim = np.mean(all_sim)
    std_sim = np.std(all_sim)
    print('Mean simi: {}, std simi:{}'.format(mean_sim, std_sim))
    return np.mean(all_sim), np.std(all_sim)

mean_sim_l1, std_sim_l1 = bootstrap_similarity(feats_before_finetune['l1'],feats_after_finetune['l1'])
mean_sim_l2, std_sim_l2 = bootstrap_similarity(feats_before_finetune['l2'],feats_after_finetune['l2'])
mean_sim_l3, std_sim_l3 = bootstrap_similarity(feats_before_finetune['l3'],feats_after_finetune['l3'])
mean_sim_l4, std_sim_l4 = bootstrap_similarity(feats_before_finetune['l4'],feats_after_finetune['l4'])
```
