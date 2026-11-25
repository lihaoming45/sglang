unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

MODEL_PATH=/mnt/share/weight/DeepSeek-V4.1-Flash-1017


export PYTHONPATH=/home/l00993641/sglang/python:$PYTHONPATH
# /flush_cache 等管理接口需要 admin key, 与服务端 SGLANG_ADMIN_API_KEY 保持一致
export SGLANG_ADMIN_API_KEY=${SGLANG_ADMIN_API_KEY:-sglang-admin}

python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 8001 \
  --model "$MODEL_PATH" \
  --tokenizer "$MODEL_PATH" \
  --dataset-name random \
  --dataset-path /mnt/share/y00627667/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenize-prompt \
  --random-input-len 65536 \
  --random-output-len 1024 \
  --random-range-ratio 1 \
  --num-prompts 64 \
  --max-concurrency 32 \
  --request-rate inf \
  --warmup-requests 0 \
  --flush-cache \
  --seed 100
