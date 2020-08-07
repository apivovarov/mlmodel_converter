#!/bin/env bash

set +e

BASENAME=`basename $(pwd)`
BASENAME="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo $BASENAME

TF1_path="${HOME}/pyenv/TF1"

echo "Framework: $FRAMEWORK"

echo "Trying default env"
python3 $BASENAME/tf_to_coreml.py
status=$?
echo "Default env exit code: $status"

if [ $status -eq 0 ]; then
    exit 0
fi

# Got Known Error. Send error to user
if [ $status -eq 4 ]; then
    echo "User Error: $status"
    exit 4
fi

# Generic error. If not TENSORFLOW then stop
if [ $FRAMEWORK != "TENSORFLOW" ]; then
    echo "Generic Error: $status"
    exit 1
fi

echo "Trying TF1"
cd $TF1_path
pipenv run python3 $BASENAME/tf_to_coreml.py
status=$?
echo "TF1 env exit code: $status"

if [ $status -eq 0 ]; then
    exit 0
fi

# Got Known Error. Send error to user
if [ $status -eq 4 ]; then
    echo "User Error: $status"
    exit 4
fi

echo "Generic Error: $status"
exit 1
