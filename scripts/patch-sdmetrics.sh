#! /bin/bash

# This script applies patches to the sdmetrics package installed in the virtual environment.
# It fixes a bug where pd.concat of categorical columns with different category sets
# (e.g. synthetic data containing misspelled values) causes dtype fallback to object,
# which then crashes XGBoost's enable_categorical=True.

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")" # resolves to <your-path-to-the-repo>/SynQTab/scripts/
PROJECT_ROOT_DIR="$(dirname "$SCRIPT_DIR")"   # resolves to <your-path-to-the-repo>/SynQTab/

SDMETRICS_SITE_PACKAGES="$PROJECT_ROOT_DIR/.venv/lib/python3.11/site-packages"

echo "Applying sdmetrics patches..."
cd "$SDMETRICS_SITE_PACKAGES"
patch -p1 --forward < "$SCRIPT_DIR/sdmetrics-patches/sdmetrics-concat-categorical.patch" \
    && echo -e "└❯ ✅ Successfully applied sdmetrics concat-categorical fix!" \
    || echo -e "└❯ ⚠️  Patch already applied or failed (see output above)."
