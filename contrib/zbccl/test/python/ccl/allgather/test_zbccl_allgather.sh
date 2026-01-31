#!/bin/bash

WORLD_SIZE=4
TEST_TYPE=float
CASE_NUM=16
export WORLD_SIZE=$WORLD_SIZE
export TEST_TYPE=$TEST_TYPE
export CASE_NUM=$CASE_NUM
rm -rf golden output
mkdir -p golden output
python3 ./scripts/data_gen.py $WORLD_SIZE $TEST_TYPE

torchrun --nproc-per-node $WORLD_SIZE --master-port 8777 test_zbccl_allgather.py