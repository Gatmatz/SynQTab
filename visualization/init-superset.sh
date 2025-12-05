#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- 1. Install Custom Database Drivers (psycopg2-binary) ---
echo "Installing custom requirements from requirements.txt..."

# Check if the requirements-local.txt file (mounted via docker-compose) exists
if [ -f /app/docker/requirements-local.txt ]; then
  # Use pip to install the requirements file
  pip install -r /app/docker/requirements-local.txt
  echo "Custom driver installation complete (psycopg2 installed)."
else
  echo "WARNING: /app/docker/requirements.txt not found. Skipping driver installation."
fi

# --- 2. Superset Database Initialization ---
echo "Running superset db upgrade..."
superset db upgrade

# --- 3. Create Admin User ---
echo "Creating admin user $ADMIN_USERNAME..."
# The '|| true' allows the script to continue if the user already exists
superset fab create-admin \
  --username "$ADMIN_USERNAME" \
  --firstname "$ADMIN_FIRSTNAME" \
  --lastname "$ADMIN_LASTNAME" \
  --email "$ADMIN_EMAIL" \
  --password "$ADMIN_PASSWORD" || true

# --- 4. Initialize Roles and Default Data ---
echo "Running superset init..."
superset init

# --- 5. Start the Superset Web Server ---
echo "Starting Superset on 0.0.0.0:8088..."
superset run -p 8088 -h 0.0.0.0