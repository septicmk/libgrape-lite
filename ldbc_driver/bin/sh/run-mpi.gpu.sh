#!/bin/bash
#
# Copyright 2015 Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
LOG_PATH=$1
APP_NAME=$2
FINAL_OUT=$3
# the binary is sync to ${HOME}/bin/standard/run_app
# switch to $HOME before mpirun
pushd ${HOME}
/bin/standard/client ${APP_NAME} ${FINAL_OUT} &
popd

echo $! > $LOG_PATH/executable.pid
wait $!
