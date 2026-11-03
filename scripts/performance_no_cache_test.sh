
# bench_serving（最常用）

#不带cache
python3 -m sglang.bench_serving   --dataset-name random   --backend sglang --host 127.0.0.1 --port 8001   --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1   --num-prompts 160 --max-concurrency 160   --warmup-requests 0 --dataset-path /home/sharegpt.json
