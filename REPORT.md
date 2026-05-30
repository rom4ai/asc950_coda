# DeepSeek-v4 on Ascend 950: Prefill/Decode Analysis

> Date: 2026-05-31
> Framework: PyPTO-style analytical model
> Hardware model: Ascend 950, 2 dies, 36 Cube cores, 72 Vector cores
> Model focus: DeepSeek-v4-flash, FP8 weights, BF16 compute/activations

## Summary

The analysis now separates three workloads that were previously easy to
confuse:

1. **Single-step decode**: one new token per sequence.
2. **Batched decode proxy**: many independent decode tokens processed together
   to model weight reuse and launch amortization.
3. **True prefill vs serial decode comparison**: same total tokens and same
   causal attention pairs, but different scheduling.

The key correction is that `B=256` is no longer reported as "prefill". It is a
batched decode proxy. True prefill is modeled with prompt length, causal
attention pairs, and KV-cache writes.

## Workload Definitions

### Decode

`decode_workload(batch=B, context_length=C, generated_tokens=G)` models `G`
serial decode steps. Each step runs the projection/MoE layer path for `B`
tokens, reads the current KV cache, and writes one new K/V entry per sequence.

Attention pairs:

```
B * sum(C + step + 1 for step in range(G))
```

### Prefill

`prefill_workload(batch=B, prompt_length=S)` models one prompt pass over all
`B*S` prompt tokens. The projection/MoE path runs once over the full prompt
batch, and causal attention covers all lower-triangular token pairs.

Attention pairs:

```
B * S * (S + 1) / 2
```

### Fair Comparison Rule

For `B=1, S=256`, prefill and serial decode both process:

- 256 total tokens
- 32,896 causal attention pairs
- identical simplified MLA attention FLOPs and KV-cache traffic formulas

The difference is scheduling: prefill runs one large linear pass; serial decode
runs 256 small linear passes.

## Ascend 950 Model

| Metric | Value |
| --- | --- |
| Dies | 2 |
| Clusters | 36 |
| Cube cores | 36 |
| Vector cores | 72 |
| BF16 peak | 354 TFLOP/s |
| HBM bandwidth | 2,400 GB/s |
| Roofline ridge | 147.5 FLOP/byte |
| K-tiling overhead | 5 us launch + 2 us pipeline fill per block |

## CODA HBM Savings

CODA fusion removes intermediate HBM materialization between GEMM epilogues,
residuals, RMSNorm, and SwiGLU-style paths.

| Batch | Pattern | Unfused | Fused | Saved | Saved % |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | GEMM+Residual+RMSNorm | 20.00 KB | 12.00 KB | 8.00 KB | 40.0% |
| 1 | GEMM+SwiGLU | 160.00 KB | 32.00 KB | 128.00 KB | 80.0% |
| 1 | Total per layer | 200.00 KB | 56.00 KB | 144.00 KB | 72.0% |
| 256 | GEMM+Residual+RMSNorm | 4.00 MB | 2.00 MB | 2.00 MB | 50.0% |
| 256 | GEMM+SwiGLU | 40.00 MB | 8.00 MB | 32.00 MB | 80.0% |
| 256 | Total per layer | 48.01 MB | 12.01 MB | 36.00 MB | 75.0% |

## Batched Decode Proxy

This table measures one decode pass with different batch sizes. It is useful
for studying weight reuse and launch amortization, but it is not a prefill
model.

| Batch | tok/s | BW util | HBM/token | HBM/pass |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 314.1 | 50.75% | 3.61 GB | 3.61 GB |
| 4 | 1,255.4 | 50.79% | 925.92 MB | 3.62 GB |
| 16 | 5,006.4 | 50.94% | 232.88 MB | 3.64 GB |
| 64 | 19,784.2 | 51.53% | 59.61 MB | 3.73 GB |
| 256 | 35,685.1 | 25.41% | 16.30 MB | 4.07 GB |

## Fair Prefill vs Serial Decode

Configuration: `B=1`, prompt/generated tokens `S=256`, FP8 weights, CODA
savings enabled.

| Mode | Tokens | Attention pairs | Linear passes | Linear us/layer | Attention us/layer | Model ms | tok/s | HBM/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill one prompt | 256 | 32,896 | 1 | 224.2 | 28.3 | 8.1 | 31,686.6 | 24.39 MB |
| Decode serial steps | 256 | 32,896 | 256 | 25,469.8 | 28.3 | 815.9 | 313.7 | 3.62 GB |

Interpretation:

- The comparison is token-fair and attention-fair.
- The attention term is identical by construction.
- Serial decode is much slower because it repeats the projection/MoE path,
  weight reads, and launch/K-tiling overhead for every generated token.
- Prefill amortizes the same layer path across all prompt tokens in one pass.

## Tests Added

The test suite now includes explicit prefill/decode coverage:

| Test file | Coverage |
| --- | --- |
| `tests/test_prefill_decode.py` | Token-fair prefill/decode comparison, causal attention-pair accounting, decode context handling, KV-cache read/write traffic, and prefill single-pass linear scheduling. |
| `tests/test_analyze_results.py` | Ensures `analyze.py` writes `prefill_decode_fair` to `results.json` and prints the fair comparison table. |
| `tests/test_hw_model.py` | Existing topology, K-tiling, roofline, batch traffic, and CODA savings checks. |
| `tests/test_baseline_pt.py` | Existing PyTorch reference checks for RMSNorm, residual GEMM, and SwiGLU. |

Latest local result:

```
12 passed
```

## Caveats

This is still an analytical model, not measured silicon data. The prefill/decode
attention model uses simplified MLA accounting:

- QK and AV are each modeled as `2 * KV-rank` FLOPs per causal pair.
- Softmax is approximated with a small per-pair FLOP term.
- KV-cache traffic counts K and V reads per causal pair and K/V writes per
  produced token.

The model is intended to make comparisons fair and reproducible, not to replace
vendor profiling on real hardware.

## Recommendations

1. Treat `B=256` results as batched decode proxy results unless prompt length is
   explicitly modeled.
2. Use `prefill_decode_comparison()` when comparing prefill and decode.
3. Keep prompt length, generated token count, attention pairs, and projection
   passes visible in all result tables.
4. Validate persistent-kernel and CODA claims separately for decode and prefill,
   because they stress different bottlenecks.
