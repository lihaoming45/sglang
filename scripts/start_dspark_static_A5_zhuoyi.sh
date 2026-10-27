#!/bin/bash
set -e

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

#export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1

export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_INTRA_DMA_ENABLE=1

export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=160
export DEEPEP_HCCL_BUFFSIZE=1600

export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_FAST_KERNEL=0
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/:$ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH

export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1

export MODEL_PATH=/mnt/share/weight/DeepSeek-V4-Flash-0731
#export PYTHONPATH="/home/x00937188/sglang-dsv4-kvquant/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH=/home/l00993641/sglang/python:$PYTHONPATH


python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --page-size 128 \
  --tp-size 8 \
  --trust-remote-code \
  --attention-backend dsv4 \
  --device npu \
  --watchdog-timeout 9000 \
  --host 0.0.0.0 --port 8001 \
  --mem-fraction-static 0.72 \
  --max-running-requests 64 \
  --chunked-prefill-size 65536 \
  --max-prefill-tokens 65536 \
  --cuda-graph-bs-decode 1 2 4 8 16 \
  --kv-cache-dtype auto \
  --enable-dp-lm-head \
  --disable-radix-cache \
  --enable-dp-attention --dp-size 8 \
  --reasoning-parser deepseek-v4 \
  --speculative-algorithm DSPARK \
  --speculative-draft-attention-backend ascend \
  --speculative-num-draft-tokens 6 \
  --speculative-dspark-block-size 5