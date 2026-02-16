#! /bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT_DIR="$(dirname "$SCRIPT_DIR")"

REQUIREMENTS_FILE="${PROJECT_ROOT_DIR}/requirements.txt"
TEMP_REQUIREMENTS="${PROJECT_ROOT_DIR}/temp_requirements.txt"

# Copy requirements.txt, removing tabpfn-extensions and adding tabpfn + tabebm
grep -v '^tabpfn-extensions' "$REQUIREMENTS_FILE" > "$TEMP_REQUIREMENTS"
echo "tabpfn==2.1.3" >> "$TEMP_REQUIREMENTS"
echo "tabebm" >> "$TEMP_REQUIREMENTS"

uv pip install --no-cache --quiet -r "$TEMP_REQUIREMENTS"
rm -f "$TEMP_REQUIREMENTS"

# Install synqtab in editable mode without resolving deps again,
# since they were already installed from the modified requirements above.
uv pip install --no-cache --quiet --no-deps -e "${PROJECT_ROOT_DIR}"

echo "========= INFO ========="
echo "Installed synqtab:" $(uv pip show synqtab | grep -i version)
echo "Installed tabpfn:" $(uv pip show tabpfn | grep -i version)
echo "Installed tabebm:" $(uv pip show tabebm | grep -i version)
