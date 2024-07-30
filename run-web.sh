#!/bin/bash

pkill python
pkill python3

# Start API server
python demo-server.py &

# Start Frontend server
/usr/bin/python3 demo-viewer.py &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
