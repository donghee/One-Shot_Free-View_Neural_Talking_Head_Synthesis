#!/bin/bash

pkill python
pkill python3
pkill frpc

python -m pip install -r docker/requirements.txt
#python -m pip install -r requirements_20240909.txt

# Start API server
python server.py &

# Start Frontend server
apt-get remove python3-urllib3 -y
# use python 3.8
/usr/bin/python3 -m pip install gradio
/usr/bin/python3 viewer.py &
# use python 3.12 for gradio5 that support streaming
#~/.pyenv/versions/3.12.4/bin/python -m pip install gradio
#~/.pyenv/versions/3.12.4/bin/python viewer.py &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
