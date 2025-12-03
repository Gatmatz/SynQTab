#!/bin/bash
set -e

# Initialize database
superset db upgrade

# Create admin user
superset fab create-admin \
  --username "$ADMIN_USERNAME" \
  --firstname "$ADMIN_FIRSTNAME" \
  --lastname "$ADMIN_LASTNAME" \
  --email "$ADMIN_EMAIL" \
  --password "$ADMIN_PASSWORD" || true

# Initialize roles + data
superset init

# Start
superset run -p 8088 -h 0.0.0.0
