#!/bin/bash

pkill python
pkill python3
pkill frpc

python -m pip install -r docker/requirements.txt
#python -m pip install -r requirements_20240909.txt

# Start API server
python demo-server.py &

# Start Frontend server
apt-get remove python3-urllib3 -y
/usr/bin/python3 -m pip install gradio
/usr/bin/python3 demo-viewer.py &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
