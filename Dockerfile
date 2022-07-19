FROM python:3.7.13-slim AS base

ENV LANG C.UTF-8

RUN apt update && \
    apt upgrade && \
    apt install -qy \
    bash-completion \
    sudo \
    curl

RUN mkdir /etc/bash_completion.d

RUN useradd -ms /bin/bash worker
RUN usermod -a -G sudo worker
RUN printf "\nworker ALL=(ALL) NOPASSWD:ALL\n" >> /etc/sudoers

USER 1000

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

RUN sudo sh -c "$HOME/.poetry/bin/poetry completions bash > /etc/bash_completion.d/poetry.bash-completion"
RUN printf "\n. ~/.poetry/env\n" >> ~/.bashrc

WORKDIR /home/worker/minidalle2
RUN python -m venv venv
RUN venv/bin/pip install --no-cache-dir -U pip
RUN venv/bin/pip install --no-cache-dir wheel

COPY --chown=worker:worker requirements.txt .
ARG OPT_PIP_EXTRA_INDEX_PYTORCH
ENV OPT_PIP_EXTRA_INDEX_PYTORCH=$OPT_PIP_EXTRA_INDEX_PYTORCH
RUN venv/bin/pip install --no-cache-dir -r requirements.txt $OPT_PIP_EXTRA_INDEX_PYTORCH

COPY --chown=worker:worker pyproject.toml .
RUN sudo sh -c "$PWD/venv/bin/poe _bash_completion > /etc/bash_completion.d/poe.bash-completion"

COPY --chown=worker:worker src src
COPY --chown=worker:worker bin bin
CMD ["tail", "-f", "/dev/null"]
