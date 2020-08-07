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

if [ $status -ne 0 ] && [ $FRAMEWORK = "TENSORFLOW" ]; then
  echo "Trying TF1"
  cd $TF1_path
  pipenv run python3 $BASENAME/tf_to_coreml.py
  status=$?
  echo "TF1 env exit code: $status"
fi

