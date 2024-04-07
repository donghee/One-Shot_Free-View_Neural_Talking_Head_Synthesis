#!/bin/bash
set -e
eval "$(pyenv init -)"
exec "$@"
