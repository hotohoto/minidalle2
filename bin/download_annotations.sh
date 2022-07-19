#!/bin/bash

PROJECT_PATH="$( cd -- "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P )"

mkdir -p $PROJECT_PATH/datasets
cd $PROJECT_PATH/datasets
wget https://www.dropbox.com/s/cqtdpsl4hewlli1/redcaps_v1.0_annotations.zip
unzip redcaps_v1.0_annotations.zip
