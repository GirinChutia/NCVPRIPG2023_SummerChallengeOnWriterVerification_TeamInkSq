## Solution code for *Summer Challenge on Writer Verification (NCVPRIPG'23)*

#### Training
> python train.py --epochs 10 --experiment_folder_path exp0 --learning_rate 0.02 --save_after 10 --batch_size 8 --train_dataset_folder training_folder_path --val_dataset_folder validation_folder_path
* --train_dataset_folder (The folder should follow the structure as shown in "Train and Validation folder structure" section)
* --val_dataset_folder (The folder should follow the structure as shown in "Train and Validation folder structure" section)
---
#### Evaluation & Generation of submission csv file



---

#### Train and Validation folder structure
```
train/
    M0021/
        A0.jpg
        A1.jpg
        A2.jpg
        ..
    M0023/
        A0.jpg
        A1.jpg
        B0.jpg
        ..
    ..
```

```
val/
    M0025/
        A0.jpg
        A1.jpg
        A2.jpg
        ..
    M1002/
        A0.jpg
        A1.jpg
        ...
    ..
```