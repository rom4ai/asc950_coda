#!/usr/bin/env python3
"""
DeepSeek-v4-pro Performance Analysis on Ascend 950 (PyPTO framework)
Compares v4-pro (d=7168) vs v4-flash (d=2048)
"""
import sys
sys.path.insert(0, ".")

from hw_model import Ascend950, DeepSeekV4FlashConfig as FlashCfg
from coda_pypto import catalog_dpsk_v4_flash, DSv4FlashConfig


def format_flops(f: float) -> str:
    if f >= 1e12: return f"{f/1e12:.2f} TFLOP"
    if f >= 1e9:  return f"{f/1e9:.2f} GFLOP"
    if f >= 1e6:  return f"{f/1e6:.2f} MFLOP"
    return f"{f/1e3:.1f} KFLOP"


def fmt_kb(b: float) -> str:
    if b >= 1e6: return f"{b/1e6:.1f} GB"
    if b >= 1e3: return f"{b/1e3:.0f} MB"
    return f"{b:.0f} KB"


class DeepSeekV4ProConfig:
    """DeepSeek-v4-pro architecture parameters (public estimates)"""
    hidden: int = 7168          # ~7K hidden (same as V3)
    q_rank: int = 1536          # MLA Q compression
    kv_rank: int = 512          # MLA KV compression
    num_experts: int = 256
    top_k: int = 8
    ffn: int = 2048             # per-expert FFN hidden
    shared_ffn: int = 2048
    num_layers: int = 61        # same as V3
    vocab_size: int = 129280
    eps: float = 1e-6
    head_dim: int = 128
    num_heads: int = 56         # 7168 / 128
    num_kv_heads: int = 8


def estimate_kernel_hbm(hidden, q_rank, kv_rank, ffn, num_experts, top_k, B=1):
    """Estimate HBM traffic for DeepSeek-v4-pro/pro kernel operations."""
    FP8, BF16 = 1, 2
    
    def kb(n, b=BF16): 
        return n * b / 1024
    
    kernels = {
        "Q_proj+RMSNorm": {
            "flops": 5*hidden + 2*hidden*q_rank,
            "read_kb": kb(B*hidden) + kb(hidden) + kb(hidden*q_rank, FP8),
            "write_kb": kb(B*q_rank),
            "unfused_kb": kb(B*hidden) + kb(hidden) + kb(hidden*q_rank, FP8) + kb(B*q_rank) + kb(B*hidden),
        },
        "KV_proj": {
            "flops": 2*hidden*2*kv_rank,
            "read_kb": kb(B*hidden) + kb(2*hidden*kv_rank, FP8),
            "write_kb": kb(B*kv_rank) + kb(B*kv_rank),
            "unfused_kb": 2*(kb(B*hidden) + kb(hidden*kv_rank, FP8) + kb(B*kv_rank)),
        },
        "AttnOut+Residual+RMSNorm": {
            "flops": 2*q_rank*hidden + B*hidden + 5*hidden,
            "read_kb": kb(B*q_rank) + kb(q_rank*hidden, FP8) + kb(B*hidden) + kb(hidden),
            "write_kb": kb(B*hidden),
            "unfused_kb": kb(B*q_rank) + kb(q_rank*hidden, FP8) + kb(B*hidden) + kb(B*hidden) + kb(B*hidden) + kb(hidden),
        },
        "MoE_Gate": {
            "flops": 2*hidden*num_experts,
            "read_kb": kb(B*hidden) + kb(hidden*num_experts, FP8),
            "write_kb": kb(B*top_k) + kb(B*top_k),
            "unfused_kb": kb(B*hidden) + kb(hidden*num_experts, FP8) + kb(B*num_experts),
        },
        "Expert_SwiGLU": {
            "flops": 2*hidden*2*ffn + 3*ffn,
            "read_kb": kb(B*hidden) + kb(2*hidden*ffn, FP8),
            "write_kb": kb(B*ffn),
            "unfused_kb": kb(B*hidden) + kb(hidden*ffn, FP8) + kb(B*ffn) + kb(B*hidden) + kb(hidden*ffn, FP8) + kb(B*ffn),
        },
        "Expert_Down+Residual": {
            "flops": 2*ffn*hidden + B*hidden,
            "read_kb": kb(B*ffn) + kb(ffn*hidden, FP8) + kb(B*hidden),
            "write_kb": kb(B*hidden),
            "unfused_kb": kb(B*ffn) + kb(ffn*hidden, FP8) + kb(B*hidden) + kb(B*hidden) + kb(B*hidden),
        },
        "Shared_Expert": {
            "flops": 2*hidden*2*ffn + 3*ffn + 2*ffn*hidden,
            "read_kb": kb(B*hidden) + kb(2*hidden*ffn, FP8) + kb(ffn*hidden, FP8),
            "write_kb": kb(B*hidden),
            "unfused_kb": kb(B*hidden) + kb(2*hidden*ffn, FP8) + kb(B*ffn) + kb(B*ffn) + kb(ffn*hidden, FP8) + kb(B*hidden),
        },
    }
    return kernels


def main():
    hw = Ascend950()
    peak_flops = hw.peak_bf16_tflops * 1e12
    peak_bw = hw.hbm_bandwidth_gb_s * 1e9
    ridge = peak_flops / peak_bw

    # Both configs
    pro = DeepSeekV4ProConfig()
    flash = DeepSeekV4ProConfig()
    flash.hidden = 2048
    flash.q_rank = 1536  # keep MLA ranks
    flash.num_layers = 32
    flash.num_heads = 16

    configs = [
        ("DeepSeek-v4-pro  (H=7168, L=61)", pro),
        ("DeepSeek-v4-flash (H=2048, L=32)", flash),
    ]

    print("=" * 115)
    print("  DeepSeek-v4 Performance on Ascend 950 (PyPTO Roofline Model)")
    print("=" * 115)
    print(f"  Hardware: {hw.peak_bf16_tflops:.0f} TFLOP/s BF16 | {hw.hbm_bandwidth_gb_s} GB/s HBM | Ridge {ridge:.0f} FLOP/B")
    print(f"  K-tiling: {hw.launch_overhead_us}us launch + {hw.pipeline_fill_us}us fill/block")
    print()

    for name, cfg in configs:
        print(f"  {'='*115}")
        print(f"  {name}")
        print(f"  H={cfg.hidden}  Q-rank={cfg.q_rank}  KV-rank={cfg.kv_rank}  FFN={cfg.ffn}")
        print(f"  MoE: {cfg.num_experts} experts, top-{cfg.top_k}  |  Layers: {cfg.num_layers}")
        print()
        
        kernels = estimate_kernel_hbm(
            cfg.hidden, cfg.q_rank, cfg.kv_rank, cfg.ffn, 
            cfg.num_experts, cfg.top_k,
        )
        
        # ── Per-kernel ──
        print(f"  {'Kernel':<28s} {'FLOPs':>9s} {'Fused':>7s} {'Unfused':>7s} {'Saved':>7s} {'Cmp(us)':>7s} {'Mem(us)':>7s} {'K-ovhd':>6s} {'Total':>7s} {'Util%':>5s}")
        print(f"  {'-'*28} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*5}")
        
        total_flops = 0
        total_time = 0
        total_hbm = 0
        total_saved = 0
        n_expert_kernels = 0
        
        for kname, k in kernels.items():
            fused_kb = k["read_kb"] + k["write_kb"]
            unfused_kb = k["unfused_kb"]
            saved_kb = unfused_kb - fused_kb
            
            compute_us = k["flops"] / peak_flops * 1e6
            memory_us = fused_kb * 1024 / peak_bw * 1e6
            # K-tiling: estimate blocks needed
            k_blocks = 1  # base
            if "Q_proj" in kname or "AttnOut" in kname or "Expert" in kname:
                # Larger models need more K-tiling
                k_dim = cfg.hidden if "Q_proj" in kname or "Gate" in kname else (
                    cfg.q_rank if "AttnOut" in kname else cfg.ffn
                )
                # K-tiling needed if k_dim > what fits in L0B (32K elements)
                k_blocks = max(1, (k_dim + 31) // 32)
            overhead_us = k_blocks * (hw.launch_overhead_us + hw.pipeline_fill_us)
            
            actual_us = max(compute_us, memory_us) + overhead_us
            achieved = k["flops"] / (actual_us * 1e-6)
            util = achieved / peak_flops * 100
            
            is_expert = "Expert" in kname
            mult = cfg.top_k if is_expert else 1
            
            total_flops += k["flops"] * mult
            total_time += actual_us * mult
            total_hbm += fused_kb * mult
            total_saved += saved_kb * mult
            if is_expert:
                n_expert_kernels += mult
            
            print(f"  {kname:<28s} {format_flops(k['flops']):>9s} {fmt_kb(fused_kb):>7s} {fmt_kb(unfused_kb):>7s} {fmt_kb(saved_kb):>7s} {compute_us:>6.2f} {memory_us:>6.2f} {overhead_us:>5.1f} {actual_us:>6.2f} {util:>4.1f}%")
        
        # Layer total
        layer_flops_non_exp = sum(k["flops"] for n,k in kernels.items() if "Expert" not in n)
        layer_flops_exp = sum(k["flops"] for n,k in kernels.items() if "Expert" in n)
        layer_flops = layer_flops_non_exp + cfg.top_k * layer_flops_exp
        
        layer_compute = layer_flops / peak_flops * 1e6
        # Full layer with all experts
        layer_time = total_time
        layer_achieved = layer_flops / (layer_time * 1e-6)
        layer_util = layer_achieved / peak_flops * 100
        
        print(f"  {'-'*28} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*5}")
        print(f"  {'LAYER (all experts)':<28s} {format_flops(layer_flops):>9s} {'':>7s} {'':>7s} {fmt_kb(total_saved):>7s} {'':>7s} {'':>7s} {'':>6s} {total_time:>6.2f} {layer_util:>4.1f}%")
        
        # ── Model-level throughput ──
        print(f"\n  {'Model Throughput (weight reuse, activations scale with B)':<80s}")
        print(f"  {'B':>4s} {'FLOPs':>9s} {'HBM':>7s} {'Time':>8s} {'tok/s':>8s} {'Util%':>6s} {'Bound':>8s}")
        print(f"  {'-'*4} {'-'*9} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
        
        for B in [1, 4, 16, 64, 256]:
            # Weights loaded once, activations scale
            # Estimate: ~5% of HBM is activation at B=1
            act_ratio = 0.05
            model_hbm_b1 = total_hbm * cfg.num_layers
            model_hbm = model_hbm_b1 * (1 - act_ratio) + model_hbm_b1 * act_ratio * B
            
            model_flops = layer_flops * cfg.num_layers * B
            
            model_compute = model_flops / peak_flops * 1e6
            model_memory = model_hbm * 1024 / peak_bw * 1e6
            model_overhead = total_time * 0.8 * cfg.num_layers  # ~80% is overhead
            
            model_time = max(model_compute, model_memory) + model_overhead
            achieved = model_flops / (model_time * 1e-6)
            util = achieved / peak_flops * 100
            tok_s = B / (model_time * 1e-6)
            
            bound = "COMPUTE" if model_compute > model_memory else "MEMORY"
            print(f"  {B:>4d} {format_flops(model_flops):>9s} {fmt_kb(model_hbm*1024):>7s} {model_time/1e3:>7.2f}ms {tok_s:>7.0f} {util:>5.1f}% {bound:>8s}")
        
        print()
    
    # ── Comparison ──
    print(f"  {'='*115}")
    print(f"  v4-pro vs v4-flash Comparison")
    print(f"  {'='*115}")
    
    # Quick comparison table
    for B in [1, 16, 256]:
        print(f"\n  B={B}:")
        print(f"  {'Metric':<20s} {'v4-flash':>20s} {'v4-pro':>20s} {'Ratio':>10s}")
        print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*10}")
        
        for model_name, cfg in configs:
            k = estimate_kernel_hbm(cfg.hidden, cfg.q_rank, cfg.kv_rank, cfg.ffn, cfg.num_experts, cfg.top_k, B)
            non_exp_f = sum(x["flops"] for n,x in k.items() if "Expert" not in n)
            exp_f = sum(x["flops"] for n,x in k.items() if "Expert" in n)
            lf = non_exp_f + cfg.top_k * exp_f
            mf = lf * cfg.num_layers * B
            
            non_exp_hbm = sum(x["read_kb"]+x["write_kb"] for n,x in k.items() if "Expert" not in n)
            exp_hbm = sum(x["read_kb"]+x["write_kb"] for n,x in k.items() if "Expert" in n)
            lhbm = non_exp_hbm + cfg.top_k * exp_hbm
            act_ratio = 0.05
            mhbm = lhbm * cfg.num_layers * (1 - act_ratio + act_ratio * B)
            
            mc = mf / peak_flops * 1e6
            mm = mhbm * 1024 / peak_bw * 1e6
            mo = lhbm / (lhbm + 1) * 0.8 * cfg.num_layers * (1 if B==1 else B*0.3)
            mt = max(mc, mm) + mo
            ts = B / (mt * 1e-6)
            util = (mf / (mt * 1e-6)) / peak_flops * 100
            
            tag = "flash" if "flash" in model_name else "pro"
            if tag == "flash" and B == 1:
                flash_flops_b1 = mf
                flash_time_b1 = mt
                flash_ts_b1 = ts
                flash_hbm_b1 = mhbm
                flash_util_b1 = util
            elif tag == "pro" and B == 1:
                pro_flops_b1 = mf
                pro_time_b1 = mt
                pro_ts_b1 = ts
                pro_hbm_b1 = mhbm
                pro_util_b1 = util
        
        # Now print comparison
        print(f"  {'FLOPs/pass':<20s} {format_flops(flash_flops_b1):>20s} {format_flops(pro_flops_b1):>20s} {pro_flops_b1/flash_flops_b1:>9.1f}x")
        print(f"  {'Time/pass':<20s} {flash_time_b1/1e3:>19.1f}ms {pro_time_b1/1e3:>19.1f}ms {pro_time_b1/flash_time_b1:>9.1f}x")
        print(f"  {'tok/s':<20s} {flash_ts_b1:>20.0f} {pro_ts_b1:>20.0f} {flash_ts_b1/pro_ts_b1:>9.1f}x")
        print(f"  {'HBM/pass':<20s} {fmt_kb(flash_hbm_b1*1024):>20s} {fmt_kb(pro_hbm_b1*1024):>20s} {pro_hbm_b1/flash_hbm_b1:>9.1f}x")
        print(f"  {'Utilization':<20s} {flash_util_b1:>19.1f}% {pro_util_b1:>19.1f}% {'':>10s}")


if __name__ == "__main__":
    main()
