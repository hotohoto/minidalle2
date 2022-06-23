#!/bin/bash
echo setup_trainer.sh running...

echo $PWD
python --version
apt-get install -qy python3.7-venv
python -m venv venv
. venv/bin/activate
which python
which pip

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
. $HOME/.poetry/env
poetry --version

pip install -U pip
if nvidia-smi; then
    echo "GPU looks available"
    pip install -r requirements_trainer.txt
else
    echo "GPU doesn't look available"
    pip install -r requirements_trainer.txt --extra-index-url https://download.pytorch.org/whl/cpu
fi
