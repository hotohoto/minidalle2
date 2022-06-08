#!/bin/bash
echo train_clip.sh running...

. bin/setup_trainer.sh
poe train_clip "$@"
