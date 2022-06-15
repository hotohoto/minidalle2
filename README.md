# Mini Dall-E 2 (WIP)

## TODO

- check if it continues training
- Read cited papers at https://github.com/lucidrains/DALLE2-pytorch
- train the model and debug it
- deploy it as an ONNX model for web browsers

## Install dependencies

```
pip install -U pip
poetry config repositories.pytorch https://download.pytorch.org/whl/cpu
poetry install
```

## Prepare datasets

Refer to this script to download images in `./datasets/images`.

```sh
cd ..
git clone git@github.com:hotohoto/redcaps-downloader.git
cd redcaps-downloader
pyenv virtualenv 3.8.12 redcaps
pyenv local redcaps
pip install -r requirements.txt

wget https://www.dropbox.com/s/cqtdpsl4hewlli1/redcaps_v1.0_annotations.zip?dl=1
unzip redcaps_v1.0_annotations.zip

for ann_file in annotations/*.json; do
    echo $ann_file
    redcaps download-imgs -a $ann_file --save-to datasets/images --resize 64 -j 16;
done
ln -s $PWD/datasets/images ../mini-dalle2/datasets/images
ln -s $PWD/annotations ../mini-dalle2/datasets/annotations
```

## Build index

```sh
poe build_index
```

## Launch your servers

```sh
poe start_servers
```

## Train remotely

In Colab, you may run commands as follow.

```
!pip install --quiet mlflow

!MLFLOW_TRACKING_URI="https://your_tracking_server_url" \
DATASETS_URL="https://your_datasets_server_url" \
mlflow run https://github.com/hotohoto/minidalle2.git -e train_clip -v main -P n-epochs=1 --env-manager=local

!MLFLOW_TRACKING_URI="https://your_tracking_server_url" \
DATASETS_URL="https://your_datasets_server_url" \
mlflow run https://github.com/hotohoto/minidalle2.git -e train_clip_reset -v main -P n-epochs=1 --env-manager=local

!MLFLOW_TRACKING_URI="https://your_tracking_server_url" \
DATASETS_URL="https://your_datasets_server_url" \
mlflow run https://github.com/hotohoto/minidalle2.git -e train_prior -v main -P n-epochs=1 --env-manager=local

!MLFLOW_TRACKING_URI="https://your_tracking_server_url" \
DATASETS_URL="https://your_datasets_server_url" \
mlflow run https://github.com/hotohoto/minidalle2.git -e train_prior_reset -v main -P n-epochs=1 --env-manager=local

!MLFLOW_TRACKING_URI="https://your_tracking_server_url" \
DATASETS_URL="https://your_datasets_server_url" \
mlflow run https://github.com/hotohoto/minidalle2.git -e train_decoder -v main -P n-epochs=1 --env-manager=local

!MLFLOW_TRACKING_URI="https://your_tracking_server_url" \
DATASETS_URL="https://your_datasets_server_url" \
mlflow run https://github.com/hotohoto/minidalle2.git -e train_decoder_reset -v main -P n-epochs=1 --env-manager=local
```

## Run tests

```bash
pytest
```

## References

- https://github.com/lucidrains/DALLE2-pytorch
- https://redcaps.xyz/
- https://github.com/redcaps-dataset/redcaps-downloader
- https://python-poetry.org/docs/repositories/
- https://github.com/hotohoto/python-example-project
- https://medium.com/analytics-vidhya/build-a-machine-learning-laboratory-from-anywhere-google-colab-ngrok-ca7590777bd8
