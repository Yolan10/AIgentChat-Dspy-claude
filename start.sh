#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

# Ensure templates directory exists
mkdir -p templates

# Initialize the database with admin user
echo "Initializing database..."
python -c "from api import init_user_db; init_user_db()"

# Start the application with gunicorn
echo "Starting application..."
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT api:app
