#!/bin/bash
rm -rf ./logs
export ASCEND_PROCESS_LOG_PATH=./logs
export ASCEND_GLOBAL_LOG_LEVEL=3

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

python test_intranode.py --num-processes=8 --num-tokens=2048 --num-topk=8 --num-experts=256
