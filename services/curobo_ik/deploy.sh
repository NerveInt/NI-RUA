#!/bin/bash
# Copy this service and the packaged SO-101 geometry to a GPU host (run from the NI-RUA checkout on the Mac).
#   bash services/curobo_ik/deploy.sh <gpu-host> '~/ni-rua-services/curobo_ik'
# The remote directory holds only this service: code, model_src/ (geometry_model.zip + physical.json), the
# generated model/ and run/. Nothing else on the host is touched.
set -eu
HOST=${1:?ssh host}; DEST=${2:?remote directory}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$HERE/../../ni_rua/robot/model/robot_information_data/so101
ssh -o ControlPath=${SSH_CONTROL_PATH:-none} "$HOST" "mkdir -p $DEST/model_src"
scp -q -o ControlPath=${SSH_CONTROL_PATH:-none} "$HERE"/curobo_ik_service.py "$HERE"/build_model.py "$HERE"/start.sh "$HERE"/stop.sh "$HERE"/README.md "$HOST:$DEST/"
scp -q -o ControlPath=${SSH_CONTROL_PATH:-none} "$SRC/geometry_model.zip" "$SRC/physical.json" "$HOST:$DEST/model_src/"
echo "deployed to $HOST:$DEST"
