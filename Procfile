web: python start.py && gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 api:app
