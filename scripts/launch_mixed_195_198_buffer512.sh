#!/usr/bin/env bash
set -euo pipefail
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
unset SGLANG_PP_OUTPUT_VIA_CPU GLM52_SYNC_WEIGHT_LOAD GLM52_STAGE_WEIGHT_COPY GLM52_DIRECT_CHECKPOINT_PATH GLM52_DIRECT_WEIGHT_NAMES GLM52_CHECKPOINT_MANIFEST
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export ACL_DEVICE_SYNC_TIMEOUT=60 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32 ASCEND_USE_FIA=1 SGLANG_NPU_USE_MLAPO=1
export TRANSFORMERS_VERBOSITY=error HCCL_HOST_SOCKET_PORT_RANGE=auto
export GLOO_SOCKET_IFNAME=data0.3001 HCCL_SOCKET_IFNAME=enp35s0f2
export DEEPEP_HCCL_BUFFSIZE=512 SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
unset HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE
# IPS=("141.61.81.109" "141.61.81.110")
IPS=("141.61.49.195" "141.61.49.198")

LOCAL_IPS=" $(hostname -I) "
for i in "${!IPS[@]}"; do
 if [[ "$LOCAL_IPS" == *" ${IPS[$i]} "* ]]; then
  echo "GLM52 mixed nnodes=2 tp16 dp16 rank=$i host=${IPS[$i]} HCCL=$HCCL_OP_EXPANSION_MODE GLOO=$GLOO_SOCKET_IFNAME"
  exec python3 -m sglang.launch_server \
    --model-path /mnt/share/w00936111/GLM-5.2-W8A8C8-mxfp8 \
    --served-model-name glm52 --reasoning-parser glm45 --tool-call-parser glm47 \
    --host "${IPS[$i]}" --port 8001 --nnodes 2 --node-rank "$i" \
    --dist-init-addr 141.61.49.195:8001 \
    --tp-size 16 --dp-size 16 --enable-dp-attention --enable-dp-lm-head \
    --model-loader-extra-config '{"enable_multithread_load":false}' \
    --load-balance-method round_robin --trust-remote-code \
    --attention-backend ascend --device npu --watchdog-timeout 9000 \
    --max-running-requests 128 --mem-fraction-static 0.92 \
    --quantization modelslim --moe-a2a-backend deepep --deepep-mode auto \
    --cuda-graph-max-bs-decode 8 --tokenizer-worker-num 16 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --dtype bfloat16 --context-length 220000 --disable-shared-experts-fusion \
    --enable-metrics
 fi
done
echo "Not a configured mixed worker: $LOCAL_IPS" >&2
exit 1
