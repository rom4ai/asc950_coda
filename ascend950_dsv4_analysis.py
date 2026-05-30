#!/usr/bin/env python3
"""
Ascend 950 Hardware Model + DeepSeek-v4-flash Performance Analysis
Based on PyPTO SoC architecture (src/backend/common/soc.cpp)
"""

import math
from dataclasses import dataclass
from typing import List, Dict

# ═══════════════════════════════════════════════════════════════════════════
# Ascend 950 Hardware Model (from PyPTO SoC definition)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Ascend950Spec:
    """Hardware spec extracted from PyPTO src/backend/common/soc.cpp"""

    num_dies: int = 2
    clusters_per_die: int = 18
    cube_cores_per_cluster: int = 1
    vec_cores_per_cluster: int = 2

    @property
    def total_clusters(self): return self.num_dies * self.clusters_per_die
    @property
    def total_cube_cores(self): return self.total_clusters * self.cube_cores_per_cluster
    @property
    def total_vec_cores(self): return self.total_clusters * self.vec_cores_per_cluster

    # AIC (Cube) core memory hierarchy (bytes)
    mat_l1_bytes: int = 512 * 1024
    left_l0a_bytes: int = 64 * 1024
    right_l0b_bytes: int = 64 * 1024
    acc_l0c_bytes: int = 256 * 1024
    bias_bytes: int = 4 * 1024

    # AIV (Vector) core memory
    vec_ub_bytes: int = 248 * 1024

    # Compute estimates
    cube_frequency_ghz: float = 1.2
    cube_mac_rows: int = 16
    cube_mac_cols: int = 16
    cube_mac_depth: int = 16

    @property
    def fp16_tflops_per_cube(self):
        return (self.cube_mac_rows * self.cube_mac_cols *
                self.cube_mac_depth * 2 * self.cube_frequency_ghz / 1000)

    @property
    def bf16_tflops_total(self):
        return self.fp16_tflops_per_cube * self.total_cube_cores

    vec_fma_per_cycle: int = 32
    vec_frequency_ghz: float = 1.2

    @property
    def fp32_tflops_total(self):
        gflops_per_vec = self.vec_fma_per_cycle * 2 * self.vec_frequency_ghz
        return gflops_per_vec * self.total_vec_cores / 1000

    # Memory bandwidth estimates (GB/s)
    hbm_bandwidth_gbps: float = 2400
    l1_bandwidth_per_core_gbps: float = 800
    l0_bandwidth_gbps: float = 3200

    # Ring buffer
    ring_slots_unidirectional: int = 8
    ring_slots_bidirectional: int = 4
    flags_per_direction: int = 8


# ═══════════════════════════════════════════════════════════════════════════
# DeepSeek-v4-flash Architecture
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeepSeekV4FlashConfig:
    hidden_size: int = 2048
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 4          # MLA KV compression
    head_dim: int = 128
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536

    num_experts: int = 256
    num_active_experts: int = 8
    expert_intermediate_size: int = 2048
    shared_expert_intermediate: int = 2048

    compute_dtype: str = "bf16"
    param_dtype: str = "fp8_e4m3"
    max_seq_len: int = 131072

    def __post_init__(self):
        self._moe_intermediate = (self.num_active_experts *
                                  self.expert_intermediate_size)

    @property
    def kv_cache_per_token_bytes(self):
        return self.num_layers * self.num_kv_heads * self.kv_lora_rank * 2

    @property
    def total_params_billions(self):
        d = self.hidden_size
        L = self.num_layers
        attn = d * (self.q_lora_rank + 2*self.kv_lora_rank) + self.q_lora_rank*d
        gate = d * self.num_experts
        experts = self.num_experts * 3 * d * self.expert_intermediate_size
        shared = 3 * d * self.shared_expert_intermediate
        per_layer = attn + gate + experts + shared
        embed = d * 128256 * 2
        return (L * per_layer + embed) / 1e9


@dataclass
class OpAnalysis:
    name: str
    flops: float
    bytes_read: float
    bytes_written: float

    @property
    def total_bytes(self): return self.bytes_read + self.bytes_written

    @property
    def arithmetic_intensity(self):
        return self.flops / self.total_bytes if self.total_bytes > 0 else 0

    def roofline_time_ms(self, hw: Ascend950Spec):
        ct = self.flops / (hw.bf16_tflops_total * 1e12)
        mt = self.total_bytes / (hw.hbm_bandwidth_gbps * 1e9)
        return max(ct, mt) * 1000

    def bottleneck(self, hw: Ascend950Spec):
        ct = self.flops / (hw.bf16_tflops_total * 1e12)
        mt = self.total_bytes / (hw.hbm_bandwidth_gbps * 1e9)
        if ct > mt:
            return "COMPUTE", ct / max(ct, mt)
        else:
            return "MEMORY", mt / max(ct, mt)


def analyze_ops(model: DeepSeekV4FlashConfig) -> List[OpAnalysis]:
    d = model.hidden_size
    fp8 = model.param_dtype == "fp8_e4m3"
    w_bytes = 1 if fp8 else 2   # weight element bytes
    a_bytes = 2                  # activation always bf16
    ops = []

    # 1. Q Projection (MLA)
    ops.append(OpAnalysis(
        "Q Proj (MLA)",
        flops=2 * d * model.q_lora_rank,
        bytes_read=d*a_bytes + d*model.q_lora_rank*w_bytes,
        bytes_written=model.q_lora_rank*a_bytes,
    ))

    # 2. KV Projection (MLA)
    ops.append(OpAnalysis(
        "KV Proj (MLA)",
        flops=2 * d * 2 * model.kv_lora_rank,
        bytes_read=d*a_bytes + 2*d*model.kv_lora_rank*w_bytes,
        bytes_written=2*model.kv_lora_rank*a_bytes,
    ))

    # 3. Attention Output
    ops.append(OpAnalysis(
        "Attn Out Proj",
        flops=2 * d * d,
        bytes_read=d*a_bytes + d*d*w_bytes,
        bytes_written=d*a_bytes,
    ))

    # 4. MoE Gate
    ops.append(OpAnalysis(
        "MoE Gate Router",
        flops=2 * d * model.num_experts,
        bytes_read=d*a_bytes + d*model.num_experts*w_bytes,
        bytes_written=model.num_experts*a_bytes,
    ))

    # 5. MoE Expert Gate/Up (single)
    ops.append(OpAnalysis(
        "MoE Expert Gate/Up (x1)",
        flops=2 * d * model.expert_intermediate_size * 2,
        bytes_read=d*a_bytes + 2*d*model.expert_intermediate_size*w_bytes,
        bytes_written=2*model.expert_intermediate_size*a_bytes,
    ))

    # 6. MoE Expert Down (single)
    ops.append(OpAnalysis(
        "MoE Expert Down (x1)",
        flops=2 * model.expert_intermediate_size * d,
        bytes_read=model.expert_intermediate_size*a_bytes + model.expert_intermediate_size*d*w_bytes,
        bytes_written=d*a_bytes,
    ))

    # 7. CODA Fused: GEMM+Residual+RMSNorm
    ops.append(OpAnalysis(
        "CODA Fused GEMM+Res+RMS",
        flops=2*d*d + 2*d,
        bytes_read=d*a_bytes + d*d*w_bytes + d*a_bytes + d*a_bytes,
        bytes_written=d*a_bytes,
    ))

    # 8. Unfused baseline
    ops.append(OpAnalysis(
        "Unfused GEMM+RMS+Res",
        flops=2*d*d + 3*d,
        bytes_read=d*a_bytes + d*d*w_bytes + d*a_bytes + d*a_bytes + d*a_bytes,
        bytes_written=d*a_bytes + d*a_bytes,
    ))

    return ops


def main():
    hw = Ascend950Spec()
    model = DeepSeekV4FlashConfig()

    # ── Hardware Summary ──
    print("=" * 70)
    print("  Ascend 950 Hardware Model (from PyPTO SoC)")
    print("=" * 70)
    ridge = hw.bf16_tflops_total * 1e12 / (hw.hbm_bandwidth_gbps * 1e9)
    print(f"""
  Topology:
    Dies:              {hw.num_dies}
    Clusters/die:      {hw.clusters_per_die} (total: {hw.total_clusters})
    Cube cores:        {hw.total_cube_cores} (1/cluster)
    Vector cores:      {hw.total_vec_cores} (2/cluster)

  AIC Cube Core Memory:          AIV Vector Core:
    Mat (L1):  {hw.mat_l1_bytes//1024:>5} KB           Vec (UB): {hw.vec_ub_bytes//1024:>5} KB
    Left (L0A):{hw.left_l0a_bytes//1024:>5} KB
    Right(L0B):{hw.right_l0b_bytes//1024:>5} KB
    Acc (L0C): {hw.acc_l0c_bytes//1024:>5} KB
    Bias:      {hw.bias_bytes//1024:>5} KB

  Compute (estimated):
    BF16:     {hw.bf16_tflops_total:.0f} TFLOPS ({hw.fp16_tflops_per_cube:.1f}/core x {hw.total_cube_cores})
    FP32 Vec: {hw.fp32_tflops_total:.1f} TFLOPS

  Memory (estimated):
    HBM BW:   {hw.hbm_bandwidth_gbps} GB/s
    Roofline ridge point: {ridge:.0f} FLOP/byte (BF16)
""")

    # ── Model Summary ──
    print("=" * 70)
    print("  DeepSeek-v4-flash Architecture")
    print("=" * 70)
    print(f"""
  Hidden:  {model.hidden_size}   Layers: {model.num_layers}   Heads: {model.num_attention_heads}Q/{model.num_kv_heads}KV
  KV latent rank: {model.kv_lora_rank}   Q latent rank: {model.q_lora_rank}
  MoE:  {model.num_experts} experts, top-{model.num_active_experts}, FFN hidden {model.expert_intermediate_size}
  Params: ~{model.total_params_billions:.0f}B   KV cache/token: {model.kv_cache_per_token_bytes/1024:.0f}KB
  Precision: {model.compute_dtype} (compute) / {model.param_dtype} (params)
""")

    # ── L0 Tile Capacity ──
    print("=" * 70)
    print("  L0 Tile Capacity Analysis (AIC Cube Core)")
    print("=" * 70)
    max_l0a = hw.left_l0a_bytes // 2
    max_l0b = hw.right_l0b_bytes // 2
    max_l0c = hw.acc_l0c_bytes // 2
    d = model.hidden_size
    print(f"  L0A: {max_l0a:,} el  L0B: {max_l0b:,} el  L0C: {max_l0c:,} el  (BF16)")
    print(f"\n  {'Operator':<24s} {'MxN':>10s} {'K':>6s} {'L0A need':>10s} {'L0B need':>10s} {'L0C':>8s} {'K-tiling':>10s}")
    print(f"  {'-'*24} {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for name, m, n, k in [
        ("Q Proj (MLA)", 1, model.q_lora_rank, d),
        ("KV Proj (MLA)", 1, model.kv_lora_rank, d),
        ("Attn Out Proj", 1, d, d),
        ("MoE Gate", 1, model.num_experts, d),
        ("MoE Gate/Up (x1)", 1, model.expert_intermediate_size, d),
        ("MoE Down (x1)", 1, d, model.expert_intermediate_size),
    ]:
        n_a = m * k
        n_b = k * n
        n_c = m * n
        kt_a = max(1, math.ceil(n_a / max_l0a))
        kt_b = max(1, math.ceil(n_b / max_l0b))
        k_blocks = max(kt_a, kt_b)
        c_ok = "OK" if n_c <= max_l0c else "EXCEEDS!"
        kt_str = f"{k_blocks}x{k//k_blocks}" if k_blocks > 1 else "1x (full)"
        print(f"  {name:<24s} {m:>3}x{n:<5} {k:>6} {n_a:>10,} {n_b:>10,} {c_ok:>8} {kt_str:>10}")

    # ── Operator Performance Analysis ──
    print(f"\n{'='*70}")
    print("  Operator Performance Analysis (per token, single layer, BF16)")
    print(f"{'='*70}")
    ridge = hw.bf16_tflops_total * 1e12 / (hw.hbm_bandwidth_gbps * 1e9)
    ops = analyze_ops(model)

    header = f"  {'Operator':<28s} {'FLOPs':>8s} {'Bytes':>8s} {'AI':>6s} {'Time(us)':>8s} {'Bound':>8s} {'Util':>6s}"
    print(header)
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")

    total_flops = 0
    total_bytes = 0
    for op in ops:
        bn, util = op.bottleneck(hw)
        t_us = op.roofline_time_ms(hw) * 1000
        total_flops += op.flops
        total_bytes += op.total_bytes
        print(f"  {op.name:<28s} {op.flops:>8.0f} {op.total_bytes:>8.0f} {op.arithmetic_intensity:>5.0f} {t_us:>7.1f} {bn:>8s} {util*100:>5.0f}%")

    total_ai = total_flops / total_bytes if total_bytes > 0 else 0
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'Layer Total':<28s} {total_flops:>8.0f} {total_bytes:>8.0f} {total_ai:>5.0f}")

    total_time_us = sum(op.roofline_time_ms(hw) * 1000 for op in ops)
    tok_per_sec = 1e6 / total_time_us if total_time_us > 0 else 0
    print(f"\n  Single-token forward time: {total_time_us:.0f} us → {tok_per_sec:.0f} tok/s (roofline bound)")
    print(f"  Ridge point: {ridge:.0f} FLOP/byte — below = MEMORY bound")

    # ── CODA Fusion Analysis ──
    print(f"\n{'='*70}")
    print("  CODA Fusion: GEMM+Residual+RMSNorm Benefit")
    print(f"{'='*70}")
    wb = 1 if model.param_dtype == "fp8_e4m3" else 2
    ab = 2
    d = model.hidden_size

    # Unfused
    u_read = d*ab + d*d*wb + d*ab + d*ab + d*ab   # x+W + gemm_out+residual+gamma
    u_write = d*ab + d*ab                            # gemm_out + norm_out
    u_total = u_read + u_write

    # Fused
    f_read = d*ab + d*d*wb + d*ab + d*ab             # x+W+residual+gamma
    f_write = d*ab                                    # norm_out only
    f_total = f_read + f_write

    saved = u_total - f_total
    saved_pct = saved / u_total * 100

    print(f"""
  Unfused (2 kernels):  {u_total/1024:.0f} KB HBM traffic
    GEMM:  read x+W ({d*ab/1024:.0f}+{d*d*wb/1024:.0f}KB), write gemm_out ({d*ab/1024:.0f}KB)
    RMS:   read gemm_out+residual+gamma ({d*ab/1024:.0f}+{d*ab/1024:.0f}+{d*ab/1024:.0f}KB), write norm ({d*ab/1024:.0f}KB)

  CODA Fused (1 kernel): {f_total/1024:.0f} KB HBM traffic
    Read:  x+W+residual+gamma ({d*ab/1024:.0f}+{d*d*wb/1024:.0f}+{d*ab/1024:.0f}+{d*ab/1024:.0f}KB)
    Write: norm_out only ({d*ab/1024:.0f}KB)

  Savings: {saved/1024:.0f} KB/token ({saved_pct:.0f}%) — gemm_out never hits HBM

  Per full model ({model.num_layers} layers, 3 CODA patterns/layer):
    Total HBM saved: {model.num_layers*3*saved/1e6:.1f} MB per forward pass
""")

    # ── Bottleneck Summary ──
    print("=" * 70)
    print("  Bottleneck Summary")
    print("=" * 70)
    mem_ops = [op for op in ops if op.bottleneck(hw)[0] == "MEMORY"]
    comp_ops = [op for op in ops if op.bottleneck(hw)[0] == "COMPUTE"]
    print(f"\n  Memory-bound ops ({len(mem_ops)}):")
    for op in mem_ops:
        print(f"    - {op.name}: AI={op.arithmetic_intensity:.0f} FLOP/byte")
    print(f"\n  Compute-bound ops ({len(comp_ops)}):")
    for op in comp_ops:
        print(f"    - {op.name}: AI={op.arithmetic_intensity:.0f} FLOP/byte")

    print(f"""
  Key Insight:
    Ridge point = {ridge:.0f} FLOP/byte. Most operators fall below this,
    meaning DeepSeek-v4-flash on Ascend 950 is predominantly MEMORY-BOUND.

    CODA fusion is especially valuable: it removes intermediate
    materialization (gemm_out of {d*ab/1024:.0f}KB) that would otherwise
    be read back from HBM in the next kernel.

  Optimization priorities for Ascend 950:
    1. Apply CODA GEMM+Residual+RMSNorm fusion (saves ~{saved_pct:.0f}% HBM traffic per pattern)
    2. FP8 weight loading (halves weight bytes vs BF16) — already used
    3. K-tiling essential for large projections (MoE gate, Q proj)
    4. Cross-cluster load balancing for MoE expert dispatch
    5. Ring buffer pipelining on A5 (Cube→Vector via SRAM, no GM round-trip)
""")


if __name__ == "__main__":
    main()
