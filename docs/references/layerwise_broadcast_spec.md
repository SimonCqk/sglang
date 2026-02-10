# Layer-wise Broadcast with Computation Overlap

## Overview

This document describes the implementation of layer-wise broadcast loading for SGLang. This feature enables faster time-to-first-token during model scale-out by allowing inference to start on early layers while later layers are still being transferred.

## Background

### Problem
When provisioning new hosts in LLM serving systems, the entire model weights must be transferred before inference can start. For large models, this can take significant time, delaying the time-to-first-token.

### Solution
Layer-wise transfer decomposes the model into layers and transfers them sequentially. The key insight is that LLM inference naturally processes layer-by-layer, so we can interleave computation and transfer - performing inference on layer N while layer N+1 is being transferred.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Seed Instance                             │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │   ModelRunner   │───▶│ send_weights_to_remote_instance_ │   │
│  │                 │    │ layerwise()                       │   │
│  └─────────────────┘    └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    │ NCCL Broadcast
                                    │ (layer by layer)
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      New Instance                                │
│  ┌──────────────────────────┐    ┌─────────────────────────┐   │
│  │ LayerwiseBroadcastModel  │───▶│  LayerReadinessTracker  │   │
│  │ Loader                   │    │                         │   │
│  └──────────────────────────┘    └─────────────────────────┘   │
│                                              │                   │
│                                              ▼                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    LlamaModel.forward()                   │   │
│  │  wait_for_embed() ──▶ wait_for_layer(i) ──▶ wait_for_lm_head()│
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Per-Layer Data Format and Transfer Order

### Transfer Order

Weights are transferred in the following strict order to enable early inference:

```
1. embed_tokens     (embedding layer - required for first forward pass)
2. layers[0]        (first transformer layer)
3. layers[1]        (second transformer layer)
   ...
N. layers[N-1]      (last transformer layer)
N+1. norm           (final normalization layer)
N+2. lm_head        (output projection - required for token generation)
```

### Per-Layer Parameter Structure (LlamaModel)

#### Embedding Layer

| Parameter Name | Shape | Description |
|---------------|-------|-------------|
| `model.embed_tokens.weight` | `[vocab_size, hidden_size]` | Token embeddings |

#### Each Transformer Layer (i = 0..N-1)

| Parameter Name | Shape | Description |
|---------------|-------|-------------|
| `model.layers.{i}.self_attn.qkv_proj.weight` | `[q_size + 2*kv_size, hidden_size]` | Merged Q/K/V projection |
| `model.layers.{i}.self_attn.o_proj.weight` | `[hidden_size, num_heads * head_dim]` | Output projection |
| `model.layers.{i}.mlp.gate_up_proj.weight` | `[2 * intermediate_size, hidden_size]` | Merged gate + up projection |
| `model.layers.{i}.mlp.down_proj.weight` | `[hidden_size, intermediate_size]` | Down projection |
| `model.layers.{i}.input_layernorm.weight` | `[hidden_size]` | Pre-attention RMSNorm |
| `model.layers.{i}.post_attention_layernorm.weight` | `[hidden_size]` | Pre-MLP RMSNorm |

#### Final Layers

| Parameter Name | Shape | Description |
|---------------|-------|-------------|
| `model.norm.weight` | `[hidden_size]` | Final RMSNorm |
| `lm_head.weight` | `[vocab_size, hidden_size]` | Output projection |

### HuggingFace to SGLang Weight Name Mapping

SGLang merges certain weights for efficiency:

| HuggingFace Weights | SGLang Weight | Notes |
|--------------------|---------------|-------|
| `q_proj`, `k_proj`, `v_proj` | `qkv_proj` | Stacked along dim 0 |
| `gate_proj`, `up_proj` | `gate_up_proj` | Stacked along dim 0 |

### Parameter Grouping

Parameters are grouped by layer prefix for ordered transfer:
- `embed_tokens`: All embedding-related parameters
- `layer_{i}`: All parameters matching `layers.{i}.*` pattern
- `norm`: Final normalization layer (excluding layer-internal norms)
- `lm_head`: Output projection parameters
- `other`: Any remaining model-specific parameters

### Model Size Estimates

| Model | Layers | Per-Layer Size | Embed/LM_head | Total |
|-------|--------|---------------|---------------|-------|
| Llama-3.1-8B | 32 | ~430 MB | ~1 GB | ~15 GB |
| Llama-3.1-70B | 80 | ~1.7 GB | ~2 GB | ~140 GB |
| Llama-3.1-405B | 126 | ~6.4 GB | ~8 GB | ~810 GB |

### Quantized Model Considerations

For quantized models (FP8, INT4, AWQ, GPTQ), additional scale tensors are transferred per linear layer.

---

## Inference and Transfer Parallelism

### Synchronization Mechanism

The parallelism is achieved through a thread-safe `LayerReadinessTracker` that coordinates between:
- **Transfer Thread**: Receives weights via NCCL and marks layers as ready
- **Inference Thread**: Waits for required layers before computation

### LayerReadinessTracker API

| Method | Called By | Purpose |
|--------|-----------|---------|
| `mark_embed_ready()` | Transfer thread | Signal embedding layer is loaded |
| `mark_layer_ready(layer_idx)` | Transfer thread | Signal specific layer is loaded |
| `mark_lm_head_ready()` | Transfer thread | Signal lm_head/norm are loaded |
| `wait_for_embed(timeout)` | Inference thread | Block until embedding ready |
| `wait_for_layer(layer_idx, timeout)` | Inference thread | Block until layer ready |
| `wait_for_lm_head(timeout)` | Inference thread | Block until lm_head ready |
| `get_max_ready_layer()` | Any | Get highest contiguously loaded layer |
| `is_fully_loaded()` | Any | Check if all weights are loaded |

### Forward Pass Integration

The model's `forward()` method integrates with `LayerReadinessTracker`:

1. Before embedding computation: `wait_for_embed()`
2. Before each layer computation: `wait_for_layer(i)`
3. Before final output: `wait_for_lm_head()`
4. If `layer_tracker` is None (normal loading), no blocking occurs

### Thread Timing Diagram

```
Time ──────────────────────────────────────────────────────────────────►

Transfer Thread:
┌──────────┬──────────┬──────────┬──────────┬─────┬──────────┬─────────┐
│  recv    │  recv    │  recv    │  recv    │ ... │  recv    │  recv   │
│  embed   │ layer_0  │ layer_1  │ layer_2  │     │  norm    │ lm_head │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴─────┴────┬─────┴────┬────┘
     │          │          │          │                │          │
     ▼          ▼          ▼          ▼                ▼          ▼
   mark_      mark_      mark_      mark_                       mark_
   embed_     layer_     layer_     layer_                      lm_head_
   ready()    ready(0)   ready(1)   ready(2)                    ready()

Inference Thread:
     │          │          │          │                           │
     ▼          ▼          ▼          ▼                           ▼
   wait &     wait &     wait &     wait &          ...        wait &
   compute    compute    compute    compute                    compute
   embed      layer_0    layer_1    layer_2                    output
```

### Computation Overlap Benefits

```
Without Layerwise (Sequential):
├── Transfer ALL layers ──────────────────────────┤├── Inference ──┤
Total Time = Transfer Time + Inference Time

With Layerwise (Overlapped):
├── Transfer embed ──┤
                     ├── Compute embed ──┤
├────── Transfer layer_0 ──────┤
                               ├── Compute layer_0 ──┤
...
Total Time ≈ max(Transfer Time, Inference Time) + overhead
```

---

## API Definitions

### External API (No Changes)

The user-facing inference API remains unchanged. Layer-wise loading is an internal optimization.

### Server Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--load-format` | str | `auto` | Set to `layerwise_remote_instance` for layerwise loading |
| `--remote-instance-weight-loader-seed-instance-ip` | str | None | IP of seed instance |
| `--remote-instance-weight-loader-seed-instance-service-port` | int | None | Service port of seed instance |
| `--remote-instance-weight-loader-send-weights-group-ports` | List[int] | Auto-allocated | NCCL group ports for each TP rank (auto-allocated if not specified) |
| `--layerwise-broadcast-timeout` | float | 60.0 | Per-layer timeout in seconds |

#### Port Auto-Allocation

When using NCCL backend for `remote_instance` or `layerwise_remote_instance` load formats, the `--remote-instance-weight-loader-send-weights-group-ports` option can be omitted. The system will automatically allocate one port per TP rank using available ports on the system. This simplifies deployment by eliminating the need to manually specify ports.

Example log output when ports are auto-allocated:
```
INFO: Auto-allocated NCCL group ports for remote instance weight loader: [29500, 29501, 29502, 29503]
```

### HTTP Endpoints

#### Initialize Weight Send Group

```
POST /init_weights_send_group_for_remote_instance
```

| Field | Type | Description |
|-------|------|-------------|
| `master_address` | str | IP address of the seed instance |
| `ports` | str | Comma-separated ports for each TP rank |
| `group_rank` | int | Rank in communication group (0=sender, 1=receiver) |
| `world_size` | int | Total world size (typically 2) |
| `group_name` | str | Name of the communication group |
| `backend` | str | Communication backend ("nccl") |

#### Send Weights (Layerwise)

```
POST /send_weights_to_remote_instance_layerwise
```

| Field | Type | Description |
|-------|------|-------------|
| `master_address` | str | IP of target instance |
| `ports` | str | Comma-separated ports for each TP rank |
| `group_name` | str | Name of the communication group |

**Response**: `{success: bool, message: str}`

### Internal Module Communication

```
HTTP Server
    │
    ▼ (async)
TokenizerManager
    │
    ▼ (ZMQ IPC)
Scheduler
    │
    ▼
TPWorker
    │
    ▼
ModelRunner.send_weights_to_remote_instance_layerwise()
    │
    ▼ (NCCL broadcast per layer)
Remote Instance
```

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `LayerReadinessTracker` | `model_loader/layer_readiness_tracker.py` | Thread-safe layer loading status tracker |
| `LayerwiseBroadcastModelLoader` | `model_loader/loader.py` | Model loader with layerwise streaming |
| `SendWeightsToRemoteInstanceLayerwiseReqInput` | `managers/io_struct.py` | Request struct for layerwise transfer |

### LoadConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `load_format` | LoadFormat | Set to `LAYERWISE_REMOTE_INSTANCE` |
| `remote_instance_weight_loader_seed_instance_ip` | str | Seed instance IP |
| `remote_instance_weight_loader_seed_instance_service_port` | int | Seed instance port |
| `remote_instance_weight_loader_send_weights_group_ports` | List[int] | Ports per TP rank |
| `layerwise_broadcast_timeout` | float | Per-layer timeout (default: 60.0s) |

---

## How It Works

### Sender Side (Seed Instance)

1. Receives request at `/send_weights_to_remote_instance_layerwise` endpoint
2. Creates a **dedicated low-priority CUDA stream** for transfer operations
3. Spawns a background thread to execute the transfer (non-blocking to scheduler)
4. Groups model parameters by layer prefix
5. Sends parameters in order: embed_tokens → layers[0..N] → norm → lm_head
6. Each layer is sent via NCCL broadcast on the dedicated transfer stream
7. Waits for each layer to complete before starting the next (non-pipelined)

#### Minimizing Impact on Sender Inference

The sender instance may be actively serving inference requests during the transfer. To minimize impact on inference performance (TTFT/TPOT), the implementation uses:

**1. Dedicated Low-Priority CUDA Stream**

```
GPU Hardware Resources:
┌─────────────────────────────────────────────────────────────┐
│  Default Stream (priority=0)  → Inference kernels (SMs)     │
│  Transfer Stream (priority=-1) → NCCL broadcast (Copy Engines)│
└─────────────────────────────────────────────────────────────┘
```

- NCCL operations primarily use **Copy Engines (DMA)**, not Streaming Multiprocessors
- Inference primarily uses **SMs** for compute
- By using separate streams, transfer and inference can execute in parallel
- Low priority (`priority=-1`) ensures GPU scheduler favors inference when resources are contended

**2. Non-Pipelined Transfer Strategy**

```
Pipelined (NOT used - continuous bandwidth contention):
Transfer: [Layer0][Layer1][Layer2]...  ← No gaps, continuous impact

Non-Pipelined (used - allows inference breathing room):
Transfer: [Layer0]    [Layer1]    [Layer2]...
Inference:    [batch]     [batch]     [batch]
              ↑ gaps allow full bandwidth for inference
```

Each layer transfer completes before the next begins, creating gaps where inference has full memory bandwidth access.

**3. Background Thread Execution**

The transfer runs in a background thread, ensuring the scheduler's main event loop remains responsive to inference requests.

```
Main Thread (Scheduler):  [recv_req][run_batch][recv_req][run_batch]...
Background Thread:        [transfer layer 0][transfer layer 1]...
                          └─ Non-blocking to main thread
```

### Receiver Side (New Instance)

1. `LayerwiseBroadcastModelLoader.load_model()` is called
2. Creates `LayerReadinessTracker` and attaches it to the model
3. Builds NCCL process group with seed instance
4. Triggers layerwise transfer request to seed instance (in background thread)
5. Receives weights layer-by-layer via NCCL broadcast
6. Marks each layer ready as it's received
7. Returns model immediately (inference can start before fully loaded)

### Forward Path (Inference)

1. Model's `forward()` checks if `layer_tracker` is set
2. Before embed_tokens: `wait_for_embed()` blocks until embedding is ready
3. Before each layer: `wait_for_layer(i)` blocks until layer i is ready
4. Before lm_head: `wait_for_lm_head()` blocks until final layers are ready
5. If tracker is None (normal loading), no blocking occurs

---

## Transfer Engine Backend

### Overview

In addition to the NCCL broadcast backend, layerwise loading supports the **Mooncake Transfer Engine** for RDMA-based weight transfer. This backend is pull-based: the receiver initiates RDMA reads directly from the seed instance's GPU memory, requiring **zero GPU overhead on the seed**.

### Architecture

```
Seed Instance (Passive)                    New Instance (Active)
┌──────────────────────┐                  ┌──────────────────────────────────┐
│  GPU Memory          │                  │  LayerwiseBroadcastModelLoader   │
│  ┌────────────────┐  │   RDMA Read      │  ┌────────────────────────────┐  │
│  │ embed_tokens    │◄─┼─────────────────┼──│ 1. Submit ALL async reads  │  │
│  │ layers[0]       │◄─┼─────────────────┼──│ 2. Poll in layer order     │  │
│  │ layers[1]       │◄─┼─────────────────┼──│ 3. Mark layers ready       │  │
│  │ ...             │◄─┼─────────────────┼──│                            │  │
│  │ norm            │◄─┼─────────────────┼──│  LayerReadinessTracker     │  │
│  │ lm_head         │◄─┼─────────────────┼──│  (same as NCCL path)       │  │
│  └────────────────┘  │                  │  └────────────────────────────┘  │
└──────────────────────┘                  └──────────────────────────────────┘
```

### How It Works (Transfer Engine)

1. **Memory Registration**: The receiver registers all model weight memory with the Transfer Engine upfront via `register_memory_region()`
2. **Metadata Fetch**: The receiver fetches remote weight pointers from the seed via HTTP (`/get_remote_instance_transfer_engine_info`)
3. **Async Submit**: All layer groups are submitted for RDMA transfer simultaneously using `batch_transfer_async_read()`, maximizing RDMA pipeline utilization
4. **Ordered Polling**: Transfer completion is polled in layer order (embed → layers[0..N] → norm → lm_head) using `get_batch_transfer_status()`
5. **Layer Signaling**: As each layer's transfer completes, `LayerReadinessTracker` marks it ready, enabling inference to proceed on that layer
6. **Post-load**: `model.post_load_weights()` is called after all transfers complete

### Key Differences from NCCL Backend

| Aspect | NCCL Backend | Transfer Engine Backend |
|--------|-------------|----------------------|
| Direction | Push (seed sends) | Pull (receiver reads) |
| Seed GPU overhead | Uses CUDA stream + Copy Engines | Zero (passive RDMA) |
| Sender changes | Background thread + NCCL broadcast | None required |
| Network | TCP/IB (NCCL managed) | RDMA (mooncake managed) |
| Pipeline | Sequential per-layer | All-async submit + ordered poll |
| Ports | NCCL group ports (auto-allocated) | Transfer Engine session (ip:port) |

### Configuration

To use the Transfer Engine backend for layerwise loading:

```bash
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B \
    --port 30001 \
    --load-format layerwise_remote_instance \
    --remote-instance-weight-loader-backend transfer_engine \
    --remote-instance-weight-loader-seed-instance-ip <SEED_IP> \
    --remote-instance-weight-loader-seed-instance-service-port 30000
```

**Requirements**:
- RDMA-capable network hardware (InfiniBand or RoCE)
- mooncake package installed (`pip install mooncake`)
- Seed instance must be running with Transfer Engine enabled

### Transfer Pipeline Visualization

```
All-Async Submit (maximizes RDMA pipeline depth):
┌──────────────────────────────────────────────────────────────────────────────┐
│ submit(embed) → submit(layer_0) → submit(layer_1) → ... → submit(lm_head)  │
└──────────────────────────────────────────────────────────────────────────────┘
                    ↓ (all in-flight simultaneously)

Ordered Polling (enables inference overlap):
poll(embed)   → mark_embed_ready()   → inference can start
poll(layer_0) → mark_layer_ready(0)  → inference processes layer 0
poll(layer_1) → mark_layer_ready(1)  → inference processes layer 1
...
poll(lm_head) → mark_lm_head_ready() → first token generated
```

---

## Modified Files Summary

| Category | Files |
|----------|-------|
| **Configuration** | `configs/load_config.py`, `server_args.py` |
| **Model Loader** | `model_loader/loader.py`, `model_loader/layer_readiness_tracker.py`, `model_loader/remote_instance_weight_loader_utils.py` |
| **Model** | `models/llama.py` (added `layer_tracker` and readiness checks) |
| **API/Communication** | `entrypoints/http_server.py`, `managers/io_struct.py`, `managers/tokenizer_communicator_mixin.py`, `managers/scheduler.py`, `managers/tp_worker.py`, `model_executor/model_runner.py` |

---

## Playbook: Reproducing Layer-wise Transfer with Inference Overlap

This section provides a step-by-step guide to reproduce and verify the layer-wise transfer + inference parallelism feature.

### Prerequisites

- Two GPU machines (or two GPUs on same machine for local testing)
- Same model accessible on both machines
- Network connectivity between machines (for NCCL)
- SGLang installed with layerwise broadcast support

### Environment Setup

```bash
# On both machines
export NCCL_DEBUG=INFO           # Enable NCCL debug logging
export SGLANG_LOG_LEVEL=DEBUG    # Enable SGLang debug logging
```

### Step 1: Start Seed Instance

On **Machine A** (seed instance with model already loaded):

```bash
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B \
    --port 30000 \
    --host 0.0.0.0
```

Wait until you see:
```
INFO: The server is fired up and ready to roll!
```

### Step 2: Start New Instance with Layerwise Loading

On **Machine B** (new instance receiving weights):

```bash
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B \
    --port 30001 \
    --load-format layerwise_remote_instance \
    --remote-instance-weight-loader-seed-instance-ip <MACHINE_A_IP> \
    --remote-instance-weight-loader-seed-instance-service-port 30000 \
    --layerwise-broadcast-timeout 60.0
```

### Step 3: Observe Layer-by-Layer Loading

On Machine B, you should see logs indicating layer-by-layer loading:

```
INFO: Loading weights from remote instance using layerwise broadcast ...
DEBUG: Receiving embed_tokens weights...
INFO: Embed tokens loaded and ready
DEBUG: Receiving layer 0 weights...
INFO: Layer 0 loaded and ready
DEBUG: Receiving layer 1 weights...
INFO: Layer 1 loaded and ready
...
DEBUG: Receiving lm_head weights...
INFO: LM head and norm loaded and ready
INFO: Finished layerwise weight loading from remote instance, time used: X.XXXXs
```

### Step 4: Send Inference Request During Loading (Overlap Test)

To verify computation overlap, send an inference request **while layers are still loading**.

Open a new terminal on Machine B and run:

```bash
# Send request immediately after starting Step 2 (before loading completes)
curl -X POST http://localhost:30001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B",
        "prompt": "Hello, how are you?",
        "max_tokens": 32
    }'
```

### Step 5: Verify Overlap Behavior

**Expected behavior**:
- The inference request will block at `wait_for_embed()` until embedding is loaded
- Then proceed layer-by-layer, blocking at each `wait_for_layer(i)` until that layer is ready
- Once enough layers are loaded, inference proceeds in parallel with remaining layer transfers

**In the logs**, you should see interleaved messages:
```
INFO: Layer 5 loaded and ready
INFO: Layer 6 loaded and ready
[Inference] Processing layer 5...
INFO: Layer 7 loaded and ready
[Inference] Processing layer 6...
...
```

### Step 6: Verify Completion

After all layers are loaded, verify the server is fully functional:

```bash
# Health check
curl http://localhost:30001/health

# Normal inference
curl -X POST http://localhost:30001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B",
        "prompt": "The capital of France is",
        "max_tokens": 16
    }'
```

### Local Testing (Single Machine with Multiple GPUs)

For testing on a single machine with 2+ GPUs:

**Terminal 1** (Seed on GPU 0):
```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B \
    --port 30000
```

**Terminal 2** (New instance on GPU 1):
```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B \
    --port 30001 \
    --load-format layerwise_remote_instance \
    --remote-instance-weight-loader-seed-instance-ip 127.0.0.1 \
    --remote-instance-weight-loader-seed-instance-service-port 30000
```

### Testing with Transfer Engine Backend

For testing with the Transfer Engine (RDMA) backend:

**Terminal 1** (Seed instance):
```bash
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B \
    --port 30000 \
    --host 0.0.0.0
```

**Terminal 2** (New instance with TE layerwise loading):
```bash
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B \
    --port 30001 \
    --load-format layerwise_remote_instance \
    --remote-instance-weight-loader-backend transfer_engine \
    --remote-instance-weight-loader-seed-instance-ip <SEED_IP> \
    --remote-instance-weight-loader-seed-instance-service-port 30000
```

**Note**: Transfer Engine requires RDMA-capable hardware. For environments without RDMA, use the NCCL backend instead.

### Metrics to Observe

| Metric | Where to Find | What to Look For |
|--------|---------------|------------------|
| Per-layer load time | Server logs | `Layer X loaded and ready` timestamps |
| Total load time | Server logs | `Finished layerwise weight loading... time used: X.XXXXs` |
| Time-to-first-token | Client response | Compare with non-layerwise loading |
| NCCL transfer stats | `NCCL_DEBUG=INFO` output | Bandwidth utilization |

### Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Connection timeout | Firewall blocking NCCL ports | Open ports or use `NCCL_SOCKET_IFNAME` |
| Layer timeout | Slow network or large model | Increase `--layerwise-broadcast-timeout` |
| NCCL error | Version mismatch | Ensure same NCCL version on both machines |
| Model mismatch | Different model configs | Use identical `--model` argument |

---

## Expected Benefits

- **Reduced Time-to-First-Token**: First inference can start after embed_tokens + first few layers are loaded
- **For a 32-layer model**: If each layer takes ~1s to transfer, first inference possible in ~5s instead of ~35s
- **Computation Overlap**: GPU computes on loaded layers while network transfers remaining layers
- **Better Resource Utilization**: GPU is not idle during weight transfer

---

## Limitations

- Transfer Engine backend requires RDMA-capable hardware and the mooncake package (`pip install mooncake`)
- Only implemented for LlamaModel (other models need similar modifications)
- Requires models with clear layer boundaries (`model.layers.{idx}.*` naming)
- No activation transfer for cooperative execution across instances

---

## Future Work

### Completed

1. **Transfer Engine backend support**: RDMA-based layerwise transfer via mooncake Transfer Engine (see [Transfer Engine Backend](#transfer-engine-backend))

### Short-term

1. **Other model architectures**: Add `layer_tracker` support to Mistral, Qwen, DeepSeek, etc.
2. **Progress reporting API**: `GET /layerwise_loading_progress`
3. **Cancellation API**: `POST /cancel_layerwise_transfer`

### Medium-term

5. **Inference-aware transfer scheduling**: Pause transfer during active inference batches, resume during idle periods
6. **Prefetching on receiver**: Load layer N+K while computing layer N (requires double buffering)
7. **Metrics and observability**: Prometheus metrics for layer loading times, sender inference impact

### Long-term

8. **Cooperative execution**: Enable deployed and scaling instances to share computation during scale-out
9. **Activation transfer**: Support transferring intermediate activations between instances
10. **Dynamic layer assignment**: Automatically determine optimal layer split based on network/compute
