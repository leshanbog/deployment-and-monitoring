## VLLM

### 1. Deployment

- More details on running OpenAI-compatible server with VLLM: https://docs.vllm.ai/en/v0.6.0/serving/openai_compatible_server.html
- Run a Docker container for VLLM on a specified GPU (device 1 in this case):

```bash
docker run \
    --gpus '"device=1"' \
    --rm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 9123:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.6.3.post1 \
    --model microsoft/Phi-3-mini-4k-instruct \
    --gpu-memory-utilization 0.6 \
    --max-model-len 1024
```

Breaking down the command:
- `docker run`: VLLM inference server is running in a isolated Docker container.
- `--gpus '"device=1"'`: This tells Docker to use GPU device 0. If you have multiple GPUs, you can specify which one to use.
- `--rm`: Automatically remove the container once it exits to keep things clean.
- `-v ~/.cache/huggingface:/root/.cache/huggingface`: We mount the local Hugging Face cache inside the container. This speeds up model loading by using cached models.
- `-p 9123:8000`: Maps port 9123 on the host to port 8000 in the container. This is how we'll access the server from our machine.
- `--ipc=host`: Shares the host's IPC namespace, which can improve performance for inter-process communication.
- `vllm/vllm-openai:v0.6.3.post1`: Specifies the Docker image to use.
- `--model microsoft/Phi-3-mini-4k-instruct`: Indicates which model we want the server to load.
- `--gpu-memory-utilization 0.6`: Allocates 60% of the GPU memory to this model, which is helpful if you're running other processes or want to run multiple models.
- `--max-model-len 1024`: Model context length.

### 2. Ping VLLM

a. Query the running VLLM server to list available models

```bash
curl http://localhost:9123/v1/models | jq
# `jq` formats the JSON output for readability
```

b. Request for completions

```bash
curl http://localhost:9123/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "prompt": "What is the capital of France?",
        "max_tokens": 20,
        "temperature": 0
    }' | jq
```

c. Request for parallel completions

```bash
curl http://localhost:9123/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "prompt": "Write an essay about the history of the internet",
        "max_tokens": 500,
        "temperature": 1.1
    }' &  # `&` makes it run in the background, so that we can send another parallel request

curl http://localhost:9123/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "prompt": "Write an essay about the history of the Amazon rainforest",
        "max_tokens": 500,
        "temperature": 1.1
    }'
```

d. Send a chat-style request to VLLM, setting up a multi-turn conversation

```bash
curl http://localhost:9123/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "messages": [
            {
                "role": "system",
                "content": "You must start every response with \"Easy question! Here is my instant answer:\"."
            },
            {"role": "user", "content": "Hey! Which is bigger: 9.11 or 9.8?"}
        ],
        "temperature": 1.2,
        "n": 2
    }' | jq
```

## 3. Run benchmark

```bash
# Clone the VLLM repository to access benchmarking scripts
git clone https://github.com/vllm-project/vllm.git

# Navigate to the benchmarks directory
cd vllm/benchmarks

# Download a sample dataset for benchmarking (ShareGPT dataset)
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Install necessary Python dependencies
python3 -m pip install numpy aiohttp transformers

# Run the benchmarking script to test VLLM server performance
python3 benchmark_serving.py \
    --backend vllm \
    --model microsoft/Phi-3-mini-4k-instruct \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 2.0 \
    --num-prompts 100 \
    --port 9123 \
    --percentile-metrics ttft,tpot,e2el \
    --metric-percentiles 90,99
```

Explanation:
- `benchmark_serving.py`: The script to benchmark the server.
- `--backend vllm`: Specifies that we're testing vLLM.
- `--model`: The model we're using.
- `--dataset-name sharegpt`: Indicates the dataset type.
- `--dataset-path`: Path to our downloaded dataset.
- `--request-rate 2.0`: Sends 2 requests per second.
- `--num-prompts 100`: Total number of prompts to send.
- `--port 9123`: The port where our vLLM server is running.
- `--percentile-metrics ttft,tpot,e2el`: The metrics to calculate: time to first token, time per output token, and end-to-end latency.
- `--metric-percentiles 90,99`: The percentiles to calculate.

## 4. AWQ

1. Deploy

```bash
docker run \
    --gpus '"device=1"' \
    --rm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 9123:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.6.3.post1 \
    --model Sreenington/Phi-3-mini-4k-instruct-AWQ \
    --gpu-memory-utilization 0.6 \
    --max-model-len 1024
```

2. Benchmark

```bash
python3 benchmark_serving.py \
    --backend vllm \
    --model Sreenington/Phi-3-mini-4k-instruct-AWQ \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 2.0 \
    --num-prompts 400 \
    --port 9123 \
    --percentile-metrics ttft,tpot,e2el \
    --metric-percentiles 90,99
```
