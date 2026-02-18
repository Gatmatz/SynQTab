#! /bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")" # resolves to <your-path-to-the-repo>/SynQTab/scripts/
PROJECT_ROOT_DIR="$(dirname "$SCRIPT_DIR")"   # resolves to <your-path-to-the-repo>/SynQTab/ which is the root of the project

echo "============ DOWNLOADING REQUIRED JARS ============"
bash $SCRIPT_DIR/download-jars.sh

echo " "
echo "============ SETTING UP PYTHON PACKAGES ============"
bash $SCRIPT_DIR/setup-uv.sh
bash $SCRIPT_DIR/install-revamped-synthcity.sh

echo " "
echo "============ SETTING UP SynQTab ============"
bash $SCRIPT_DIR/install-synqtab.sh

echo " "
echo "============ PATCHING SDMETRICS ============"
bash $SCRIPT_DIR/patch-sdmetrics.sh
