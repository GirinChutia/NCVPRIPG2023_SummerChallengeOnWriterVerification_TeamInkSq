## Solution code for *Summer Challenge on Writer Verification (NCVPRIPG'23)*

---
#### Create Enviroment

Create a conda or python virtual environment (python version : 3.9) and install the packages mentioned in **_requirements.txt_** file

---

#### Training

Activate the created enviroment and run the below commands to start training the model.
Refer *_modelling/model.py_* file to know more about the model used.
> python train.py --epochs 10 --experiment_folder_path exp0 --learning_rate 0.02 --save_after 10 --batch_size 8 --train_dataset_folder training_folder_path --val_dataset_folder validation_folder_path

Args :
* **--epochs** : Total epochs for training
* **--experiment_folder_path** : Folder path to save logs and trained weights
* **--learning_rate** : Learning rate to be used for training
* **--batch_size** : Batch size to be used for training
* **--save_after** : Weight saving frequency.
* **--train_dataset_folder** : Training folder path(The folder should follow the structure as shown in "Train and Validation folder structure" section)
* **--val_dataset_folder** Validation folder path (The folder should follow the structure as shown in "Train and Validation folder structure" section)
---
#### Evaluation & Generation of submission csv file

> python perform_evaluation.py --test_csv test_csv_path --test_dir test_folder_path --model_path model_weights_path --submission_csv csv_path

Args :
* **--test_csv** : Path of val.csv or test.csv csv file (Provided in the competition)
* **--test_dir** : Path of the val or the test folder
* **--model_path** : Path of the model to be used for inference
* **--submission_csv** : Path of the csv file to which the results will be saved.


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