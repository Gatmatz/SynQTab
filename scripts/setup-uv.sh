#!/bin/bash

wget -qO- https://astral.sh/uv/install.sh | sh &> /dev/null
uv venv --quiet --no-cache --clear --python 3.11.5
source .venv/bin/activate
