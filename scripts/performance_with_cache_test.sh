
# bench_serving（最常用）

#带cache
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host 0.0.0.0 \
  --port  8001\
  --dataset-name generated-shared-prefix \
  --backend sglang \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group 150 \
  --gsp-system-prompt-len 117965 \
  --gsp-question-len 13107 \
  --gsp-output-len 1024 \
  --max-concurrency 150 \
  --random-range-ratio 1 \
  --seed 100 \
  --model $WP \
  --tokenizer $WP \
  --warmup-requests 16 \
  --request-rate 1.5 \
  --cache-report
