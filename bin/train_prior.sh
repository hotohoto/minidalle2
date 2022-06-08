#!/bin/bash
echo train_prior.sh running...

. bin/setup_trainer.sh
poe train_prior "$@"
