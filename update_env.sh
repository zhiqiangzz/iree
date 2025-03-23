#! /usr/bin/env bash

UPDATE_PATH="$(pwd)/build/Release/tools"
if [ -d "$UPDATE_PATH" ]; then
    export PATH="$UPDATE_PATH:$PATH"
else
    echo "Warning: $UPDATE_PATH does not exist!"
fi
