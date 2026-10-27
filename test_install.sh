echo "Setting up the test environment..."

if [ ! -f poetry.lock ]; then
    echo "poetry.lock file not found. Initializing poetry..."
    poetry init
    echo "poetry.lock file created"
else
    echo "poetry.lock file already exists. Re-initializing lock file..."
    poetry lock
    echo "Installing dependencies..."
    poetry install
fi

echo "Test environment setup complete."
