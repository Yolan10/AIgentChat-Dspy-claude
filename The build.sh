#!/bin/bash
# build.sh - Build script for Render deployment

echo "Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Creating logs directory..."
mkdir -p logs

echo "Build complete!"
