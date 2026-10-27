#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

echo "Setting up the test environment..."

if [ ! -f poetry.lock ]; then
    echo "poetry.lock file not found. Creating lock file..."
else
    echo "poetry.lock file already exists. Refreshing lock file..."
fi
poetry lock

echo "Installing dependencies..."
poetry install --with dev

echo "Test environment setup complete."
