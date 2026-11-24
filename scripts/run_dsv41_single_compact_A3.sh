#!/bin/bash
# Weight loading test for DeepSeek V4.1 with Engram support
# Based on run128K_v4.sh but simplified for weight loading only

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

# Kill any existing processes
pkill -9 python 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true

# Use our modified sglang code
export TORCH_EXTENSIONS_DIR="$(mktemp -d /tmp/dsv41-jit.XXXXXX)"
export CANN_OP_BUILDER_NO_FPRINT_CACHE=1

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
# source /usr/local/Ascend/ascend-toolkit/latest/vendors/dsv41a3_ transformer/bin/set_env.bash
# source /usr/local/Ascend/ascend-toolkit/latest/vendors/dsv41a3cmp_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh


export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1

#TEST
export TASK_QUEUE_ENABLE=1
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_DEFAULT_THINKING=1
#export SGLANG_DSV4_REASONING_EFFORT=max

#export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export HCCL_BUFFSIZE=128 # 1024
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# debug
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
# export ASCEND_LAUNCH_BLOCKING=1

export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048

# skip gpu branch
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

#export PYTHONPATH=/mnt/share/chenxu/dsv4_1/sglang/python:$PYTHONPATH
export PYTHONPATH=/home/zcl/sglang/python:$PYTHONPATH
export MODEL_PATH=/home/weights/DeepSeek-V4.1-Flash-1017-w8a8
#export MODEL_PATH=/home/chenxu/dsv4/

# perfermance
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
#export SGLANG_NPU_USE_MULTI_STREAM=1
#export USE_NPU_MOE_GATING_TOP_K=1


# export SGLANG_GRAPH_REPLAY_SYNC=1
# export SGLANG_DSV4_GRAPH_META_DEBUG=1
export SGLANG_DSV4_GRAPH_MAX_SEQ=131072

export SGLANG_DEFAULT_THINKING=1
#export SGLANG_DSV4_REASONING_EFFORT=max

# export ASCEND_LAUNCH_BLOCKING=1
#export ASCEND_GLOBAL_LOG_LEVEL=1
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
export SGLANG_DSV4_NATIVE_ATTN=0
# export SGLANG_DSV4_DBG_ATTN=1
export SGLANG_SHALLOW_FUSED_ROPE_FQ4=0

export SGLANG_COMPRESSOR_PREFILL_USE_TRITON=1
export SGLANG_COMPRESSOR_DECODE_USE_TRITON=1

export SGLANG_RAGGED_VERIFY_MODE=compact

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 \
    --tp-size 16 \
    --trust-remote-code \
    --attention-backend dsv4 \
    --device npu \
    --watchdog-timeout 9000 \
    --host 0.0.0.0 --port 8001 \
    --mem-fraction-static 0.96 \
    --max-running-requests 16\
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 131072 \
    --kv-cache-dtype auto \
    --enable-dp-lm-head \
    --cuda-graph-bs-decode 1 2 4 8 10 \
    --disable-radix-cache  \
    --enable-dp-attention --dp-size 8 \
    --reasoning-parser deepseek-v4 \
    --language-model-only \
    --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"], "quantization_config": null}' \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "${MODEL_PATH}" \
    --speculative-draft-model-quantization modelslim \
    --speculative-draft-kv-cache-dtype bf16 \
    --speculative-dspark-block-size 5 \
	--speculative-dspark-sps-table-path /home/l00993641/sglang/scripts/dspark_graph_additive.json \


#     --max-running-requests 16 \
# --cuda-graph-bs-decode 1 2 4 8 10 16 \
#    --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"]}'
#    --moe-a2a-backend deepep \

#        --deepep-mode auto
    # --context-length 4096 \
    # --max-total-tokens 4096 \
    #    --disable-overlap-schedule \

