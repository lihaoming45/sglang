#!/bin/bash

export PYTHONPATH=/home/l00993641/sglang/python:$PYTHONPATH

python3 -m sglang.benchmark.dspark_sps_profiler all \
    --base-url http://127.0.0.1:30100 \
    --batch-size 1 2 4 8 10 \
    --out /home/l00993641/sglang/scripts/dspark_graph_static.json \
    --no-plot