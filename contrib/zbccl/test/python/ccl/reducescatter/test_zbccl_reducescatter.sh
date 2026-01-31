#!/bin/bash

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

echo $CURRENT_DIR

WORLD_SIZE=4
TEST_TYPE=float
CASE_NUM=16
export WORLD_SIZE=$WORLD_SIZE
export TEST_TYPE=$TEST_TYPE
export CASE_NUM=$CASE_NUM
export CURRENT_DIR=$CURRENT_DIR
rm -rf golden output
mkdir -p golden output
python3 ${CURRENT_DIR}/scripts/data_gen.py $WORLD_SIZE $TEST_TYPE

torchrun --nproc-per-node $WORLD_SIZE --master-port 8777 ${CURRENT_DIR}/test_zbccl_reducescatter.py