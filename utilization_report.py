#!/usr/bin/env python3
"""
Ascend 950 Utilization Report for DeepSeek-v4-flash
Utilization = (workload_ops / runtime) / peak_performance
"""
import sys
sys.path.insert(0, ".")

from hw_model import Ascend950
from coda_pypto import catalog_dpsk_v4_flash, DSv4FlashConfig


def format_flops(f: float) -> str:
    if f >= 1e12: return f"{f/1e12:.2f} TFLOP"
    if f >= 1e9:  return f"{f/1e9:.2f} GFLOP"
    if f >= 1e6:  return f"{f/1e6:.2f} MFLOP"
    return f"{f/1e3:.1f} KFLOP"


def main():
    hw = Ascend950()
    cfg = DSv4FlashConfig()
    kernels = catalog_dpsk_v4_flash(cfg)
    
    peak_flops = hw.peak_bf16_tflops * 1e12   # FLOP/s
    peak_bw    = hw.hbm_bandwidth_gb_s * 1e9   # bytes/s
    ridge      = peak_flops / peak_bw           # FLOP/byte
    
    print("=" * 108)
    print("  Ascend 950 Utilization Report — DeepSeek-v4-flash")
    print("=" * 108)
    print(f"  Peak BF16:  {hw.peak_bf16_tflops:.0f} TFLOP/s  |  HBM BW: {hw.hbm_bandwidth_gb_s} GB/s  |  Ridge: {ridge:.0f} FLOP/B")
    print(f"  K-tiling:   {hw.launch_overhead_us}us launch + {hw.pipeline_fill_us}us fill per block")
    print(f"  Model:      H={cfg.hidden}  Q={cfg.q_rank}  KV={cfg.kv_rank}  FFN={cfg.ffn}  MoE=top-{cfg.top_k}")
    print()

    # ── FLOP estimation per kernel ──
    H, Q, KV, FFN = cfg.hidden, cfg.q_rank, cfg.kv_rank, cfg.ffn
    B = 1

    # FLOP counts (2× for multiply-add in GEMM)
    flops_map = {
        "coda_q_proj_rmsnorm":      5*H + 2*H*Q,          # RMSNorm(5H) + GEMM(2*H*Q)
        "coda_kv_proj":             2*H*(2*KV),            # GEMM: H × 2*KV
        "coda_attn_out_residual_rmsnorm": 2*Q*H + H + 5*H,  # GEMM(2*Q*H) + residual(H) + RMSNorm(5H)
        "coda_moe_gate":            2*H*cfg.num_experts,   # GEMM: H × 256
        "coda_expert_swiglu":       2*H*(2*FFN) + 5*FFN,  # GEMM + SiLU + mul
        "coda_expert_down_residual": 2*FFN*H + H,          # GEMM + residual add
        "coda_shared_expert":       2*H*(2*FFN) + 5*FFN + 2*FFN*H,  # gate/up + SwiGLU + down
    }

    print(f"  {'Operator':<35s} {'FLOPs':>10s} {'HBM r/w':>8s} {'AI':>5s} "
          f"{'Cmp(us)':>8s} {'Mem(us)':>8s} {'K-ovhd':>7s} {'Time':>8s} "
          f"{'Util%':>6s} {'Bound':>8s}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*6} {'-'*8}")

    total_flops = 0
    total_compute_us = 0
    total_memory_us = 0
    total_overhead_us = 0
    total_time_us = 0
    total_hbm = 0

    for k in kernels:
        flops = flops_map.get(k.name, 0)
        hbm_bytes = (k.hbm_read_kb + k.hbm_write_kb) * 1024
        ai = flops / hbm_bytes if hbm_bytes > 0 else float('inf')
        
        # Roofline times
        compute_us = flops / peak_flops * 1e6
        memory_us  = hbm_bytes / peak_bw * 1e6
        
        # K-tiling overhead (from the existing model)
        # Estimate k_blocks based on kernel type
        if k.name in ("coda_q_proj_rmsnorm", "coda_kv_proj", "coda_attn_out_residual_rmsnorm",
                       "coda_moe_gate", "coda_expert_swiglu", "coda_expert_down_residual",
                       "coda_shared_expert"):
            k_blocks = 1  # most fit in 1 k-tile for FP8 at these sizes
            overhead_us = k_blocks * (hw.launch_overhead_us + hw.pipeline_fill_us)
        else:
            overhead_us = 0

        actual_us = max(compute_us, memory_us) + overhead_us
        
        # Utilization = achieved FLOPs / peak FLOPs
        # achieved FLOPs = flops / actual_time
        achieved_flops = flops / (actual_us * 1e-6)
        utilization = achieved_flops / peak_flops * 100
        
        bound = "COMPUTE" if compute_us > memory_us else "MEMORY"
        if overhead_us > max(compute_us, memory_us):
            bound = "K-TILE"

        total_flops += flops
        total_compute_us += compute_us
        total_memory_us += memory_us
        total_overhead_us += overhead_us
        total_time_us += actual_us
        total_hbm += hbm_bytes

        print(f"  {k.name:<35s} {format_flops(flops):>10s} {hbm_bytes/1024:>7.0f}K {ai:>4.0f} "
              f"{compute_us:>7.2f} {memory_us:>7.2f} {overhead_us:>6.1f} {actual_us:>7.2f} "
              f"{utilization:>5.1f}% {bound:>8s}")

    # ── Layer totals ──
    non_exp_flops = sum(flops_map.get(k.name,0) for k in kernels 
                        if "expert" not in k.name and "shared" not in k.name)
    exp_flops = sum(flops_map.get(k.name,0) for k in kernels 
                    if "expert" in k.name and "shared" not in k.name)
    shared_flops = flops_map.get("coda_shared_expert", 0)
    layer_flops = non_exp_flops + cfg.top_k * exp_flops + shared_flops

    # Full layer time
    layer_compute_us = layer_flops / peak_flops * 1e6
    layer_hbm_bytes = total_hbm  # rough: per-kernel sums
    layer_memory_us = layer_hbm_bytes / peak_bw * 1e6
    
    # With 8 experts
    full_layer_hbm = sum((k.hbm_read_kb + k.hbm_write_kb) * 1024 for k in kernels)
    full_layer_hbm = full_layer_hbm * cfg.top_k  # scale experts
    full_memory_us = full_layer_hbm / peak_bw * 1e6
    
    full_compute_us = layer_flops / peak_flops * 1e6
    full_overhead = total_overhead_us * (1 + cfg.top_k)  # 8x expert overhead
    full_time = max(full_compute_us, full_memory_us) + full_overhead
    
    full_achieved = layer_flops / (full_time * 1e-6)
    full_util = full_achieved / peak_flops * 100

    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*6} {'-'*8}")
    print(f"  {'FULL LAYER (8 experts)':<35s} {format_flops(layer_flops):>10s} {'':>8s} {'':>5s} "
          f"{full_compute_us:>7.2f} {full_memory_us:>7.2f} {full_overhead:>6.1f} {full_time:>7.2f} "
          f"{full_util:>5.1f}% {'MEMORY':>8s}")

    # ── Model-level utilization ──
    print(f"\n  {'='*108}")
    print(f"  Model-Level Utilization Summary")
    print(f"  {'='*108}")
    
    for B in [1, 4, 16, 64, 256]:
        model_flops = layer_flops * cfg.num_layers * B
        model_hbm = full_layer_hbm * cfg.num_layers  # weights once, activations scale
        
        # Activation bytes scale with B (estimate: 10% of total HBM is activation at B=1)
        act_ratio = 0.10
        model_hbm_scaled = model_hbm * (1 - act_ratio) + model_hbm * act_ratio * B
        
        model_compute_us = model_flops / peak_flops * 1e6
        model_memory_us = model_hbm_scaled / peak_bw * 1e6
        model_overhead = full_overhead * cfg.num_layers
        model_time = max(model_compute_us, model_memory_us) + model_overhead
        
        achieved_flops = model_flops / (model_time * 1e-6)
        util = achieved_flops / peak_flops * 100
        
        tok_per_sec = B / (model_time * 1e-6)
        bound = "COMPUTE" if model_compute_us > model_memory_us else "MEMORY"
        
        print(f"  B={B:>3d}  FLOPs={format_flops(model_flops):>8s}  "
              f"HBM={model_hbm_scaled/1e9:.1f}GB  "
              f"Time={model_time/1e3:.2f}ms  "
              f"Util={util:>5.1f}%  "
              f"tok/s={tok_per_sec:>7.0f}  [{bound}]")

    # ── Utilization breakdown ──
    print(f"\n  {'='*108}")
    print(f"  Utilization Breakdown: Where do the FLOPs go?")
    print(f"  {'='*108}")
    
    ideal_time = layer_flops / peak_flops * 1e6  # compute-only time at peak
    actual_time = max(full_compute_us, full_memory_us) + full_overhead
    
    print(f"  Layer FLOPs:      {format_flops(layer_flops)}")
    print(f"  Ideal time:       {ideal_time:.2f} us (100% compute utilization)")
    print(f"  Memory stall:     {max(0, full_memory_us - full_compute_us):.2f} us (waiting for HBM)")
    print(f"  K-tiling overhead:{full_overhead:.2f} us (kernel launch + pipeline fill)")
    print(f"  Actual time:      {actual_time:.2f} us")
    print(f"  Compute util:     {full_compute_us / actual_time * 100:.1f}% of wall time")
    print(f"  Memory util:      {min(full_compute_us, full_memory_us) / actual_time * 100:.1f}% of wall time")
    print(f"  Overhead util:    {full_overhead / actual_time * 100:.1f}% of wall time")
    print(f"  Achieved TFLOPS:  {layer_flops / (actual_time * 1e-6) / 1e12:.1f} / {hw.peak_bf16_tflops:.0f} peak")
    print(f"  Effective util:   {layer_flops / (actual_time * 1e-6) / peak_flops * 100:.1f}%")


if __name__ == "__main__":
    main()
