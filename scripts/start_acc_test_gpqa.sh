evalscope eval \
    --model dsv \
    --api-url http://127.0.0.1:8001/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --generation-config '{
        "max_tokens": 64000,
        "seed": 3407,
        "top_p": 1.0,
        "temperature": 1.0,
        "n": 1,
        "timeout": 3600,
        "stream": false,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": true,
                "reasoning_effort": "max"
            }
        }
    }' \
    --datasets gpqa_diamond  \
    --dataset-hub local \
    --dataset-args '{
        "gpqa_diamond": {
            "local_path": "/home/gpqa_diamond",
            "eval_split": "train"
        }
    }' \
    --eval-batch-size 64 \
    --ignore-error \


    --model /home/weights/DeepSeek-V4-Flash-0731-w8a8 \



unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
unset ALL_PROXY all_proxy

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

# MODEL_PATH=/home/weights/DeepseekV4-Flash-0731-W4A8
MODEL_PATH=/home/weights/DeepSeek-V4-Pro-0813-w4a8

evalscope eval \
    --model ${MODEL_PATH} \
    --api-url http://127.0.0.1:31000/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --generation-config '{"max_tokens":120000,"top_p":1,"temperature":1,"timeout":6000,"stream":true,"extra_body":{"chat_template_kwargs":{"thinking":true}}}' \
    --datasets gpqa_diamond \
    --dataset-args '{"gpqa_diamond":{"local_path":"/home/f00447229/2026-7-30-sglang-deepseekv4-flash/dataset/gpqa","subset_list":["gpqa_diamond"]}}' \
    --eval-batch-size 32 \
    --ignore-errors