#!/bin/bash

# run servers and kill the processes running in the background
# https://stackoverflow.com/a/22644006/1874690

uvicorn --port=8000 --app-dir=src minidalle2.server.api.main:app &
trap "exit" INT TERM
trap "kill 0" EXIT

GUNICORN_CMD_ARGS="--timeout 180" mlflow server \
--backend-store-uri=sqlite:///mlflow.db \
--artifacts-destination=./mlartifacts \
--serve-artifacts
