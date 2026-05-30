# DeepSeek-v4 on Ascend 950: Performance Analysis & Optimization

> **Date:** 2026-05-31
> **Framework:** PyPTO-style analytical model
> **Hardware:** Ascend 950 (2 dies x 18 clusters, 36 Cube + 72 Vector cores)
> **Models:** DeepSeek-v4-flash (H=2048, L=32); v4-pro retained as architectural context

---

## 1. Ascend 950 Hardware Model

Extracted from the local model used by `hw_model.py`.

### Die / Cluster Topology

```
2 Dies x 18 Clusters/Die = 36 Clusters
Per Cluster: 1 AIC (Cube) + 2 AIV (Vector)
Total: 36 Cube cores + 72 Vector cores
```

### AIC Cube Core Memory

| Memory | Size | Role |
|--------|------|------|
| Mat (L1) | 512 KB | Weight/prefetch buffer |
| Left (L0A) | 64 KB | Operand A |
| Right (L0B) | 64 KB | Operand B |
| Acc (L0C) | 256 KB | Accumulator / epilogue scratch |
| Bias | 4 KB | Bias buffer |

### AIV Vector Core

| Memory | Size | Role |
|--------|------|------|
| Vec (UB) | 248 KB | Unified buffer for vector ops |

### A5 Ring Buffer

Consumer-local SRAM for Cube-to-Vector data channels. No GM round-trip is
modeled for A5 handoff.

### Compute & Bandwidth (estimated)

| Metric | Value |
|--------|-------|
| BF16 Peak | 354 TFLOP/s |
| HBM Bandwidth | 2,400 GB/s |
| Roofline Ridge | 147.5 FLOP/byte |
| K-tiling overhead | 5 us launch + 2 us pipeline fill/block |

---

## 2. DeepSeek-v4 Architecture

| Parameter | v4-flash | v4-pro | Ratio |
|-----------|----------|--------|-------|
| Hidden size | 2,048 | 7,168 | 3.5x |
| Layers | 32 | 61 | 1.9x |
| Attention | MLA: Q-rank 1536, KV-rank 512 | same | - |
| MoE | 256 experts, top-8 | 256 experts, top-8 | - |
| Expert FFN | 2,048 | 2,048 | - |
| Precision | BF16 compute / FP8 params | BF16 compute / FP8 params | - |

### v4-flash Model Complexity in Current Code

| Metric | Value |
|--------|-------|
| FLOPs/layer, B=1 | 0.242 GFLOP |
| HBM/layer after CODA, B=1 | 121.18 MB |
| Full model traffic, B=1 decode | 3.61 GB |
| Full model time, B=1 decode | 3.18 ms |
| Full model throughput, B=1 decode | 314.1 tok/s |

---

## 3. CODA Kernel Implementation (PyPTO DSL)

CODA-style fusion keeps memory-bound Transformer intermediates in GEMM
epilogues or local SRAM instead of materializing every intermediate in HBM.

### Kernel Catalog

| Kernel | Pattern | CODA Fusion |
|--------|---------|-------------|
| `coda_q_proj_rmsnorm` | RMSNorm + Q projection | Norm intermediate stays in Acc |
| `coda_kv_proj` | Packed K+V projection | Single weight load, split in epilogue |
| `coda_attn_out_residual_rmsnorm` | AttnOut + residual + RMSNorm | GEMM output avoids HBM round-trip |
| `coda_moe_gate` | MoE router + top-k | Router logits stay on-chip |
| `coda_expert_swiglu` | Gate/up GEMM + SwiGLU | Gate/up intermediates stay local |
| `coda_expert_down_residual` | Down proj + residual add | Down output consumed in epilogue |
| `coda_shared_expert` | Shared expert path | SwiGLU handoff via A5-style SRAM |
| `coda_dpsk_v4_flash_layer` | Full layer orchestrator | Composes CODA-fused sub-kernels |

### CODA HBM Savings

| Batch | Pattern | Unfused | Fused | Saved | Saved % |
|-------|---------|--------:|------:|------:|--------:|
| B=1 | GEMM+Residual+RMSNorm | 20.00 KB | 12.00 KB | 8.00 KB | 40.0% |
| B=1 | GEMM+SwiGLU | 160.00 KB | 32.00 KB | 128.00 KB | 80.0% |
| B=1 | Total per layer | 200.00 KB | 56.00 KB | 144.00 KB | 72.0% |
| B=256 | GEMM+Residual+RMSNorm | 4.00 MB | 2.00 MB | 2.00 MB | 50.0% |
| B=256 | GEMM+SwiGLU | 40.00 MB | 8.00 MB | 32.00 MB | 80.0% |
| B=256 | Total per layer | 48.01 MB | 12.01 MB | 36.00 MB | 75.0% |

> Note: these are fused-intermediate savings. Total layer HBM is still dominated
> by weight traffic at small decode batch sizes.

---

## 4. Roofline Performance Analysis

### Per-Layer Time Breakdown (v4-flash, B=1 decode)

```
Compute       0.68 us
Memory       50.49 us
K-tiling     49.00 us
TOTAL        99.49 us/layer
```

### Bottleneck

The current B=1 decode layer is memory-bound with a large K-tiling overhead
component. Peak compute is not the limiting factor.

### Per-Operator Detail (B=1, FP8 weights)

| Operator | FLOPs | HBM traffic | AI | Kblk | Total us | Bottleneck |
|----------|------:|------------:|---:|-----:|---------:|------------|
| Q_proj | 6.29 MFLOP | 3.01 MB | 2.0 | 1 | 8.314 | K-tiling |
| KV_proj | 2.10 MFLOP | 1.00 MB | 2.0 | 1 | 7.439 | K-tiling |
| AttnOut | 6.29 MFLOP | 3.01 MB | 2.0 | 1 | 8.314 | K-tiling |
| MoE_gate | 1.05 MFLOP | 516.50 KB | 2.0 | 1 | 7.220 | K-tiling |
| MoE_gate_up | 134.22 MFLOP | 64.09 MB | 2.0 | 1 | 35.003 | Memory |
| MoE_down | 67.11 MFLOP | 32.04 MB | 2.0 | 1 | 20.996 | Memory |
| RMSNorm | 10.24 KFLOP | 12.00 KB | 0.8 | 0 | 0.005 | Memory |
| Residual | 2.05 KFLOP | 12.00 KB | 0.2 | 0 | 0.005 | Memory |
| SharedExpert | 25.17 MFLOP | 12.01 MB | 2.0 | 1 | 12.246 | K-tiling |

---

## 5. Utilization Report

### Formula

```
Utilization = achieved bandwidth or achieved FLOP/s / hardware peak
```

### Batched Decode Proxy

This table is a decode proxy with different batch sizes. It is not a prefill
model because it has no prompt length or causal attention matrix.

| Batch | tok/s | BW util | HBM/token | HBM/pass |
|-------|------:|--------:|----------:|---------:|
| B=1 | 314.1 | 50.75% | 3.61 GB | 3.61 GB |
| B=4 | 1,255.4 | 50.79% | 925.92 MB | 3.62 GB |
| B=16 | 5,006.4 | 50.94% | 232.88 MB | 3.64 GB |
| B=64 | 19,784.2 | 51.53% | 59.61 MB | 3.73 GB |
| B=256 | 35,685.1 | 25.41% | 16.30 MB | 4.07 GB |

### Fair Prefill vs Serial Decode

The fair comparison holds token count and causal attention pairs fixed.

| Mode | Tokens | Attn pairs | Linear passes | KV read/layer | KV write/layer | Linear us/layer | Attn us/layer | Model ms | tok/s | HBM/token |
|------|-------:|-----------:|--------------:|--------------:|---------------:|----------------:|--------------:|---------:|------:|----------:|
| Prefill one prompt | 256 | 32,896 | 1 | 64.25 MB | 512.00 KB | 224.2 | 28.3 | 8.1 | 31,686.6 | 24.39 MB |
| Decode serial steps | 256 | 32,896 | 256 | 64.25 MB | 512.00 KB | 25,469.8 | 28.3 | 815.9 | 313.7 | 3.62 GB |

### Fairness Rule

```
Prefill attention pairs = B * S * (S + 1) / 2
Decode attention pairs  = B * sum(C + step + 1 for step in range(G))
```

For this report, `B=1`, `S=256`, `C=0`, and `G=256`, so both workloads have
the same 32,896 causal attention pairs. The difference is scheduling:
prefill uses one large projection/MoE pass, while serial decode repeats the
same path 256 times.

### Decode Context Sweep

This sweep holds decode generation length fixed at one token and varies the
existing KV-cache context length.

| Context | Tokens | Attn pairs | Linear passes | KV read/layer | KV write/layer | Attn us/layer | Model ms | tok/s |
|--------:|-------:|-----------:|--------------:|--------------:|---------------:|--------------:|---------:|------:|
| 0 | 1 | 1 | 1 | 2.00 KB | 2.00 KB | 0.002 | 3.184 | 314.1 |
| 128 | 1 | 129 | 1 | 258.00 KB | 2.00 KB | 0.111 | 3.187 | 313.7 |
| 512 | 1 | 513 | 1 | 1.00 MB | 2.00 KB | 0.439 | 3.198 | 312.7 |
| 2,048 | 1 | 2,049 | 1 | 4.00 MB | 2.00 KB | 1.749 | 3.240 | 308.7 |

### Prefill Prompt-Length Sweep

This sweep models true prompt prefill: one projection/MoE pass over all prompt
tokens plus causal attention over the prompt.

| Prompt | Tokens | Attn pairs | Linear passes | KV read/layer | KV write/layer | Linear us/layer | Attn us/layer | tok/s |
|-------:|-------:|-----------:|--------------:|--------------:|---------------:|----------------:|--------------:|------:|
| 64 | 64 | 2,080 | 1 | 4.06 MB | 128.00 KB | 101.1 | 1.8 | 19,432.5 |
| 128 | 128 | 8,256 | 1 | 16.12 MB | 256.00 KB | 136.6 | 7.2 | 27,826.9 |
| 256 | 256 | 32,896 | 1 | 64.25 MB | 512.00 KB | 224.2 | 28.3 | 31,686.6 |
| 512 | 512 | 131,328 | 1 | 256.50 MB | 1.00 MB | 399.4 | 112.5 | 31,257.9 |

---

## 6. Optimization Roadmap

### Strategy Comparison

| Strategy | Applies to | Expected effect | Status in this repo |
|----------|------------|-----------------|---------------------|
| CODA epilogue fusion | Prefill + decode | Reduces intermediate HBM traffic | Modeled and tested |
| Batched / continuous decode | Decode | Amortizes weight reads and launch overhead | Modeled as batch proxy |
| Persistent kernel | Decode + prefill | Reduces repeated launch overhead | Modeled as separate no-K sensitivity proxy |
| Prompt-length prefill modeling | Prefill | Makes prefill/decode comparison fair | Implemented and swept |
| KV-cache context sweep | Decode | Measures long-context decode cost | Implemented via `context_length` sweep |

### Persistent-Kernel Sensitivity Proxy

The no-K rows remove modeled K-tiling overhead. They are a proxy for persistent
kernel launch amortization, not measured kernels.

| Mode | Tokens | Attn pairs | Linear passes | KV read/layer | KV write/layer | Baseline ms | No-K ms | Saved ms | Baseline tok/s | No-K tok/s |
|------|-------:|-----------:|--------------:|--------------:|---------------:|------------:|--------:|---------:|---------------:|-----------:|
| decode | 1 | 1 | 1 | 2.00 KB | 2.00 KB | 3.184 | 1.616 | 1.568 | 314.1 | 618.9 |
| prefill | 256 | 32,896 | 1 | 64.25 MB | 512.00 KB | 8.079 | 6.511 | 1.568 | 31,686.6 | 39,317.3 |

### Current Optimization Interpretation

The old `B=256` row should be read as batched decode, not prefill. True prefill
is better represented by `prefill_workload(batch, prompt_length)`.

```
B=1 decode:                 314.1 tok/s
B=256 batched decode proxy: 35,685.1 tok/s
S=256 true prefill:         31,686.6 tok/s
S=256 serial decode:        313.7 tok/s
Decode no-K proxy:          618.9 tok/s
S=256 prefill no-K proxy:   39,317.3 tok/s
```

---

## 7. Key Insights

### 1. B=256 Was Misleading as "Prefill"

The numeric result is still useful, but the label was wrong. It measures a
batched decode proxy, not prompt prefill.

### 2. Fair Prefill/Decode Requires Equal Attention Work

The updated report compares prefill and serial decode with the same total
tokens and same causal attention pairs. This removes an important source of
unfairness from the old report.

### 3. Decode Is Dominated by Repetition

Serial decode repeats projection/MoE, weight reads, and K-tiling/launch overhead
for every generated token. That is why 256 serial decode steps are roughly two
orders of magnitude slower than one 256-token prefill pass in this model.

### 4. CODA Helps, But It Is Not the Whole Decode Fix

CODA reduces intermediate HBM traffic. Decode still needs batching, persistent
kernels, or another launch-amortization mechanism to avoid repeated small
kernel overhead.

### 5. Tests Now Encode the Workload Semantics

The test suite now checks the exact workload definitions, the fair
prefill/decode result block, long-context decode sweeps, prompt-length prefill
sweeps, KV-cache fields, and separated persistent-kernel sensitivity results.

---

## 8. Recommendations

| Priority | Action | Status |
|----------|--------|--------|
| P0 | Keep `B=256` labeled as batched decode proxy unless prompt length is modeled | Enforced in report wording and analysis keys |
| P0 | Use `prefill_decode_comparison()` for all prefill/decode claims | Implemented for fair comparison |
| P0 | Report tokens, attention pairs, linear passes, and KV-cache traffic in every comparison | Implemented in fair comparison and sweeps |
| P1 | Add long-context decode sweeps over `context_length` | Implemented for 0, 128, 512, and 2,048 tokens |
| P1 | Add prefill prompt-length sweeps over `prompt_length` | Implemented for 64, 128, 256, and 512 tokens |
| P2 | Validate persistent-kernel claims separately for decode and prefill | Implemented as separate no-K sensitivity proxy |

---

## Appendix: Code Locations and Test Coverage

| File | Contents |
|------|----------|
| `hw_model.py` | Ascend 950 model, roofline engine, prefill/decode workload APIs |
| `analyze.py` | Analysis entry point and `results.json` generator |
| `coda_pypto.py` | CODA-style PyPTO kernel sketches |
| `baseline_pt.py` | PyTorch unfused reference kernels |
| `results.json` | Structured benchmark and fair comparison results |
| `tests/test_hw_model.py` | Hardware, roofline, batch, and CODA tests |
| `tests/test_prefill_decode.py` | Token-fair prefill/decode and KV-cache tests |
| `tests/test_analyze_results.py` | Ensures analysis output includes fair comparison, sweeps, KV-cache fields, and persistent-kernel sensitivity |
| `tests/test_baseline_pt.py` | PyTorch reference op tests |

Latest verification:

```
python analyze.py
python -m pytest  # 13 passed
```
