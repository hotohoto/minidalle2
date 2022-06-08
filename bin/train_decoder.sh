#!/bin/bash
echo train_decoder.sh running...

. bin/setup_trainer.sh
poe train_decoder "$@"
