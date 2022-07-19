#!/bin/bash
echo update_requirements.sh running...

poetry export --without-hashes > requirements.txt
