#!/bin/bash
# start.sh - Start script with debugging

echo "=== STARTUP DEBUG INFO ==="
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

echo -e "\nChecking frontend/dist:"
if [ -d "frontend/dist" ]; then
    echo "frontend/dist exists!"
    echo "Contents:"
    ls -la frontend/dist/
else
    echo "ERROR: frontend/dist does not exist!"
fi

echo -e "\nChecking templates:"
if [ -d "templates" ]; then
    echo "templates directory exists!"
    echo "Contents:"
    ls -la templates/
else
    echo "WARNING: templates directory does not exist!"
fi

echo -e "\nPython path:"
which python

echo -e "\nPython version:"
python --version

echo -e "\nInstalled packages:"
pip list | grep -E "(flask|gunicorn|eventlet)"

echo -e "\n=== STARTING APPLICATION ==="

# Start the application
exec gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT api:app
