# Spatio-Temporal Analysis Task

## Preliminaries

- Our code framework is built based on BasicTS+. You can install PyTorch following the instruction in PyTorch. For example:
    - ```pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html```

- After ensuring that PyTorch is installed correctly, you can install other dependencies via:
    - ```pip install -r requirements.txt```

## Preparing Data

- Pre-process Data
    - ```cd /path/to/your/project```
    - ```python scripts/data_preparation/${DATASET_NAME}/generate_training_data.py```
- Replace `${DATASET_NAME}` with one of `Rainfall_UK`, `Rainfall_USA`, `Rainfall_CN`, `Rainfall_AUS`, `Rainfall_SA`, `GLOBAL`. The processed data will be placed in `datasets/${DATASET_NAME}`.

## Reproducing Built-in Models

- ```python experiments/train.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py --gpus '0'```


