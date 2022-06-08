#!/bin/bash
echo update_requirements_trainer.sh running...

poetry export --without-hashes > requirements_trainer.txt
