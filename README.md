# Mini Dall-E 2 (WIP)

## TODO

- do exploratory data analysis
  - check data quality
  - get images in a random order
- check if it continues training
- add flask/fastapi for donwloading images without index
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

```sh
mkdir -p datasets
cd datasets
wget https://www.dropbox.com/s/cqtdpsl4hewlli1/redcaps_v1.0_annotations.zip
unzip redcaps_v1.0_annotations.zip

poe import_annotations  # import annotations. this is recommended to finish this completely
poe download_images  # download images as much as you want
poe update_splits  # update the trainset index and the testset index to take account of the recently downloaded images
```

## Launch your servers

```sh
poe start_servers
```

You may want to access your local servers from a remote host.
Then you can use alternative addresses provided by ngrok.

```sh
poe start_ngrok
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
