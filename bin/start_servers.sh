#!/bin/bash

# run servers and kill the processes running in the background
# https://stackoverflow.com/a/22644006/1874690

python -m http.server --directory datasets &
trap "exit" INT TERM
trap "kill 0" EXIT

ngrok start --log=stdout --config ngrok_config.yaml --all &
trap "exit" INT TERM
trap "kill 0" EXIT

mlflow server --serve-artifacts --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
