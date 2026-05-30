#!/usr/bin/env python3
"""
CODA Performance Comparison: Ascend 950 + DeepSeek-v4-flash
Integrates hw_model roofline with coda_pypto HBM traffic catalog.
"""
import sys
sys.path.insert(0, ".")

from hw_model import Ascend950, DeepSeekV4FlashConfig
from coda_pypto import catalog_dpsk_v4_flash, DSv4FlashConfig as CodaCfg


def main():
    hw = Ascend950()
    cfg = CodaCfg()

    print("=" * 90)
    print("  CODA Performance Comparison: Ascend 950 + DeepSeek-v4-flash")
    print("=" * 90)
    print(f"  Hardware: {hw.peak_bf16_tflops:.0f} TFLOPS BF16  |  {hw.hbm_bandwidth_gb_s} GB/s HBM")
    print(f"  Ridge: {hw.ridge_flop_per_byte:.0f} FLOP/byte  |  K-tiling overhead: {hw.launch_overhead_us}+{hw.pipeline_fill_us}us/block")
    print(f"  Model: H={cfg.hidden}  Q-rank={cfg.q_rank}  KV-rank={cfg.kv_rank}  FFN={cfg.ffn}")
    print(f"  MoE: {cfg.num_experts} experts, top-{cfg.top_k}  |  Layers: {cfg.num_layers}")
    print()

    kernels = catalog_dpsk_v4_flash(cfg)
    hbm_gbps = hw.hbm_bandwidth_gb_s * 1e9

    # ── Per-kernel comparison ──
    print("  Per-Kernel HBM Traffic Comparison (B=1, FP8 weights)")
    print(f"  {'Kernel':<35s} {'Fused':>8s} {'Unfused':>8s} {'Saved':>8s} {'%':>6s} {'Time':>8s} {'Speedup':>7s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*7}")

    total_fused_kb = 0
    total_unfused_kb = 0

    for k in kernels:
        fused_kb = k.hbm_read_kb + k.hbm_write_kb
        unfused_kb = k.unfused_hbm_kb
        saved_kb = unfused_kb - fused_kb
        saved_pct = saved_kb / unfused_kb * 100 if unfused_kb > 0 else 0
        fused_us = fused_kb * 1024 / hbm_gbps * 1e6
        unfused_us = unfused_kb * 1024 / hbm_gbps * 1e6
        speedup = unfused_us / fused_us if fused_us > 0 else 1.0

        total_fused_kb += fused_kb
        total_unfused_kb += unfused_kb

        print(f"  {k.name:<35s} {fused_kb:>7.0f}K {unfused_kb:>7.0f}K {saved_kb:>7.0f}K {saved_pct:>5.1f}% {fused_us:>7.2f}u {speedup:>6.2f}x")

    # ── Full layer (8 experts) ──
    non_exp_fused = sum(k.hbm_read_kb + k.hbm_write_kb for k in kernels if "expert" not in k.name and "shared" not in k.name)
    non_exp_unfused = sum(k.unfused_hbm_kb for k in kernels if "expert" not in k.name and "shared" not in k.name)
    exp_fused = sum(k.hbm_read_kb + k.hbm_write_kb for k in kernels if "expert" in k.name and "shared" not in k.name)
    exp_unfused = sum(k.unfused_hbm_kb for k in kernels if "expert" in k.name and "shared" not in k.name)
    shared_fused = sum(k.hbm_read_kb + k.hbm_write_kb for k in kernels if "shared" in k.name)
    shared_unfused = sum(k.unfused_hbm_kb for k in kernels if "shared" in k.name)

    layer_fused = non_exp_fused + cfg.top_k * exp_fused + shared_fused
    layer_unfused = non_exp_unfused + cfg.top_k * exp_unfused + shared_unfused
    layer_saved = layer_unfused - layer_fused

    layer_us_f = layer_fused * 1024 / hbm_gbps * 1e6
    layer_us_u = layer_unfused * 1024 / hbm_gbps * 1e6

    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*7}")
    print(f"  {'FULL LAYER (8 experts)':<35s} {layer_fused:>7.0f}K {layer_unfused:>7.0f}K {layer_saved:>7.0f}K {layer_saved/layer_unfused*100:>5.1f}% {layer_us_f:>7.2f}u {layer_us_u/layer_us_f:>6.2f}x")

    # ── Multi-batch analysis ──
    print(f"\n  {'='*90}")
    print(f"  Multi-Batch Throughput Comparison (with weight reuse)")
    print(f"  {'='*90}")
    print(f"  {'B':>4s}  {'Fused tok/s':>12s}  {'Unfused tok/s':>13s}  {'Speedup':>7s}  {'HBM util':>8s}  {'CODA saved':>10s}  {'Bottleneck':>11s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*13}  {'-'*7}  {'-'*8}  {'-'*10}  {'-'*11}")

    num_layers = cfg.num_layers

    for B in [1, 4, 16, 64, 256]:
        # Per-layer traffic with weight reuse:
        # Weight bytes: read once per batch (same for all B)
        # Activation bytes: scale with B
        # We approximate: non-expert weights are the bulk, activation scales linearly
        act_bytes_b1 = layer_fused * 1024 * 0.05  # ~5% is activation at B=1
        wgt_bytes = layer_fused * 1024 - act_bytes_b1

        fused_bytes = wgt_bytes + act_bytes_b1 * B
        unfused_bytes = layer_unfused * 1024 * (0.05 * B + 0.95)  # same proportion

        fused_us = fused_bytes / hbm_gbps * 1e6
        unfused_us = unfused_bytes / hbm_gbps * 1e6

        # Full model time per token (or per batch)
        model_us_f = fused_us * num_layers
        model_us_u = unfused_us * num_layers

        tok_s_f = B / (model_us_f * 1e-6)
        tok_s_u = B / (model_us_u * 1e-6)

        hbm_util_f = (fused_bytes / (fused_us * 1e-6)) / hbm_gbps * 100
        saved_mb = (layer_unfused - layer_fused) * 1024 / 1e6 * B

        speedup = tok_s_f / tok_s_u if tok_s_u > 0 else 1.0
        bottleneck = "memory" if fused_us > (layer_fused * 1024 * 0.05 * B / (hw.peak_flops_per_s * 1e-6)) else "compute"

        print(f"  {B:>4d}  {tok_s_f:>12.0f}  {tok_s_u:>13.0f}  {speedup:>6.2f}x  {hbm_util_f:>7.1f}%  {saved_mb:>9.1f}MB  {bottleneck:>11s}")

    # ── Absolute HBM savings across model ──
    print(f"\n  {'='*90}")
    print(f"  Full Model CODA Savings Summary")
    print(f"  {'='*90}")

    for B in [1, 4, 16, 64, 256]:
        per_layer_saved_kb = layer_unfused - layer_fused
        # Adjust: activation portion scales with B
        act_saved_kb = per_layer_saved_kb * 0.5  # ~half the savings are activation-related
        wgt_saved_kb = per_layer_saved_kb - act_saved_kb
        total_saved_kb_per_layer = wgt_saved_kb + act_saved_kb * B
        total_saved_mb = total_saved_kb_per_layer * num_layers / 1024

        fused_total_kb = (wgt_bytes/1024 + act_bytes_b1/1024 * B) * num_layers * (layer_fused/layer_unfused)
        pct = total_saved_kb_per_layer / (fused_total_kb/num_layers + total_saved_kb_per_layer) * 100 if fused_total_kb > 0 else 0

        print(f"  B={B:>3d}: {total_saved_mb:>8.1f} MB saved across {num_layers} layers  "
              f"({total_saved_kb_per_layer:.0f} KB/layer)")


if __name__ == "__main__":
    main()
