#!/bin/bash

# A5 机型: 单节点 8 卡, tp8 / dp8
# 依据 A5 上验证过的 V4.1 启动脚本改写; DSPARK 投机, 图模式关闭

MODEL_PATH=/mnt/share/weight/DeepSeek-V4.1-Flash-1017
#SGLANG_PATH=$HOME/repos/sglang/python

# ===== CPU / System =====
if [[ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
  echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
fi
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING || true

# ===== Ascend Environment =====
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export TRANSFORMERS_VERBOSITY=error

# 单机不需要; auto 依赖 /etc/hccl_rootinfo.json, 文件损坏时 HCCL 建链报 error code 4
# export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

export SGLANG_SET_CPU_AFFINITY=1
export TASK_QUEUE_ENABLE=1
export INF_NAN_MODE_FORCE_DISABLE=1

export SGLANG_DEFAULT_THINKING=1
#export SGLANG_DSV4_REASONING_EFFORT=max

export HCCL_BUFFSIZE=400
export DEEPEP_HCCL_BUFFSIZE=1800

# ===== Ascend Toolkit =====
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
if [[ -f /usr/local/memfabric_hybrid/set_env.sh ]]; then
  source /usr/local/memfabric_hybrid/set_env.sh
fi

# ===== Model / SGLang Environment =====
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# ===== DSPARK =====
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_FAST_KERNEL=0

export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=150

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

# 参考脚本里的开关; 本分支 Python 侧没有对应 env 定义, 由自定义算子包读取
export SGLANG_DSV4_NATIVE_ATTN=0
export SGLANG_DSV4_DBG_ATTN=1
export SGLANG_SHALLOW_FUSED_ROPE_FQ4=0
export SGLANG_COMPRESSOR_PREFILL_USE_TRITON=1
export SGLANG_COMPRESSOR_DECODE_USE_TRITON=1
export SGLANG_OPT_USE_HC_TRITON=0
export SGLANG_OPT_USE_HC_TILELANG=0

export SGLANG_DSV4_GRAPH_MAX_SEQ=131072

# /start_profile 等管理接口是 ADMIN_FORCE: 服务端没配 admin key 时一律 403
export SGLANG_ADMIN_API_KEY=${SGLANG_ADMIN_API_KEY:-sglang-admin}

# cannbot DSL 的 L2 磁盘缓存; 不设只有进程内缓存, 8 个 rank 各编一遍(单 variant 约 172s)
#export CANNBOTDSL_CACHE_DIR=$HOME/.cache/cannbot_cache

export CANNBOTDSL_CACHE_DIR=/mnt/share/l00850654/code/dsv41/cannbot_cache

export PYTHONPATH=/home/l00993641/sglang/python:$PYTHONPATH

export PREFILL_DECODE_INTERVAL=8
# ===== Launch =====
# chunked-prefill-size 会被 attn-dp-size 整除 (parallel_hook.py: 4096 // 8 = 512),
# 那是每个 rank 一步 prefill 的 token 数; max-prefill-tokens 不除, 是每 rank 的批上限.
# 单 rank prefill 约 1.6k tok/s, 所以每步停顿 ≈ chunk / 8 / 1600 秒:
#   131072 -> 16384/rank -> ~10s   (之前实测 Max ITL 10.8s)
#    32768 ->  4096/rank -> ~2.5s
#    16384 ->  2048/rank -> ~1.3s
#     4096 ->   512/rank -> ~0.3s
#
# PREFILL_DECODE_INTERVAL: 每跑完一个 prefill batch, 强制连跑 N 轮 decode 再调度下一个
# prefill. 默认关. 长输入 (64k+) 且高并发时, decode 会被别人的 prefill 打断, TPOT 涨到
# 10 倍; 把它开起来能把 decode 的时间占比抬上去, 代价是 TTFT 变长.
# 从 8 开始试; 4/8/16 各跑一遍同一份压测, 看 TPOT 和 TTFT 的交换比.
EXTRA_ARGS=()
if [[ -n "${PREFILL_DECODE_INTERVAL:-}" ]]; then
  EXTRA_ARGS+=(--prefill-decode-interval "$PREFILL_DECODE_INTERVAL")
fi

python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name dsv4 \
    --host 0.0.0.0 --port 8001 \
    --tp-size 8 \
    --page-size 128 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --attn-dp-size 8 \
    --enable-dp-lm-head \
    --watchdog-timeout 9000 \
    --max-running-requests 64 \
    --mem-fraction-static 0.82 \
    --max-prefill-tokens 131072 \
    --chunked-prefill-size 65536 \
    --kv-cache-dtype auto \
    --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"]}' \
    --reasoning-parser deepseek-v4 \
    --language-model-only \
    --moe-a2a-backend deepep --deepep-mode auto \
    --enable-metrics \
    --speculative-algorithm DSPARK \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 6 \
    --speculative-dspark-block-size 5 \
    --cuda-graph-bs-decode 1 2 4 8 10 16 24  \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
