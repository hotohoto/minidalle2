#!/bin/bash

# run servers and kill the processes running in the background
# https://stackoverflow.com/a/22644006/1874690

python -m http.server --directory datasets &
trap "exit" INT TERM
trap "kill 0" EXIT

mlflow server \
--backend-store-uri=sqlite:///mlflow.db \
--default-artifact-root=./mlartifacts
