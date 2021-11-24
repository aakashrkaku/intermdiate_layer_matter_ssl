# Intermdiate layer matters - SSL
The official repository for "Intermediate Layers Matter in Momentum Contrastive Self Supervised Learning" paper [[pdf](https://openreview.net/pdf?id=M5j42PvY65V)].

![image](https://user-images.githubusercontent.com/32464452/143272414-231278ab-a114-4372-9ace-f0beba8bf6bb.png)

## Summary of the paper
1. Bringing intermediate layers’ representations of two augmented versions of an image closer together helps to improve the momentum contrastive (MoCo) method
2. We show this improvement for two loss functions: the mean squared error (MSE) and Barlow Twin’s loss between the intermediate layer representations; and three datasets: NIH-Chest Xrays, Breast Cancer Histopathology, and Diabetic Retinopathy 
3. Improved MoCo has large gains (~5%) in the performance especially when we are in a low-labeled regime (1% data is labeled)
4. Improved MoCo learns meaningful features earlier in the model and also has high feature reuse.

## Use the code to reproduce results
1. Download the data for the experiments:

The data can be downloaded from kaggle.com. 
NIH chest-xray dataset: https://www.kaggle.com/nih-chest-xrays/data
Breast cancer histopathology dataset: https://www.kaggle.com/paultimothymooney/breast-histopathology-images
Diabetic Retinopathy dataset: https://www.kaggle.com/c/diabetic-retinopathy-detection/data

2. Training of SSL models:

To train the ssl models for moco, moco-mse and moco-btwins, please use 'train_ssl_moco.py', 'train_ssl_moco_mse.py' and 'train_ssl_moco_btwins.py' respectively. The code works for first two datasets. For the diabetic retinopathy dataset, please write a dataloader like "chest_xray_supervised.py" and a datamodule file like "chest_xray_dm.py". Import these files in 'train_ssl_moco.py', 'train_ssl_moco_mse.py' and 'train_ssl_moco_btwins.py' and make necesary changes. The same code can work for the diabetic retinopathy dataset.

3. Fine tuning the models:

To finetune the models, please use the "fine_tune_moco_chestxray.py" and "fine_tune_moco_hist.py" for NIH chest xray and Breast cancer histopathology data, respectively. For the diabetic retinopathy dataset, please write the code for fine tuning using/similar to "fine_tune_moco_chestxray.py"

4. Probing the models:

To probe the intermediate layers of the model, please use the "probing_moco_chestxray.py" and "probing_moco_hist.py" for NIH chest xray and Breast cancer histopathology data, respectively. For the diabetic retinopathy dataset, please write the code for probing the intermediate layers using/similar to "probing_moco_chestxray.py"

5. Feature reuse analysis:

To compute the feature similarity, perform the inference using your model, store the intermediate layer representations and use "CKA.py" for computing the kernel similarity with sigma = 0.8.
