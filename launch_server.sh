export NCCL_DEBUG=INFO           # Enable NCCL debug logging
export SGLANG_LOG_LEVEL=DEBUG    # Enable SGLang debug logging


python -m sglang.launch_server     --model Qwen/Qwen3-4B-Thinking-2507-FP8    --host 0.0.0.0     --port 30000