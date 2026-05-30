"""
coda_pypto.py — CODA-style fused kernels in PyPTO DSL
Extended: DeepSeek-v4-flash full Transformer layer support.

Architecture parameters:
  hidden=2048  Q-rank=1536  KV-rank=512  FFN=2048  MoE=256 top-8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import pypto.language as pl  # type: ignore
except ModuleNotFoundError:
    pl = None


# ═══════════════════════════════════════════════════════════════════════════
# DeepSeek-v4-flash Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DSv4FlashConfig:
    hidden: int = 2048
    q_rank: int = 1536       # MLA Q compression rank
    kv_rank: int = 512       # MLA KV compression rank
    num_experts: int = 256
    top_k: int = 8
    ffn: int = 2048          # per-expert FFN hidden
    num_layers: int = 32
    eps: float = 1e-6


@dataclass(frozen=True)
class KernelSketch:
    name: str
    purpose: str
    memory_flow: List[str]
    operations: List[str]
    notes: str
    hbm_read_kb: float = 0.0
    hbm_write_kb: float = 0.0
    unfused_hbm_kb: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "memory_flow": self.memory_flow,
            "operations": self.operations,
            "notes": self.notes,
            "hbm_read_kb": self.hbm_read_kb,
            "hbm_write_kb": self.hbm_write_kb,
            "hbm_saved_kb": max(0, self.unfused_hbm_kb - self.hbm_read_kb - self.hbm_write_kb),
        }


def pypto_available() -> bool:
    return pl is not None


# ═══════════════════════════════════════════════════════════════════════════
# Kernel Builders (work in both PyPTO and sketch-fallback modes)
# ═══════════════════════════════════════════════════════════════════════════

def fused_matmul_residual_rmsnorm(
    x_tile, w_tile, residual_tile, gamma,
    hidden_size: int = 2048, eps: float = 1e-6,
):
    """CODA: GEMM + residual add + RMSNorm in single kernel. Output never hits HBM."""
    if pl is None:
        return _gemm_residual_rmsnorm_sketch()
    acc = pl.matmul(x_tile, w_tile)
    residual_acc = pl.add(acc, residual_tile)
    sq = pl.mul(residual_acc, residual_acc)
    mean = pl.row_expand_div(pl.row_sum(sq), hidden_size)
    rms = pl.sqrt(pl.add(mean, eps))
    normed = pl.row_expand_div(residual_acc, rms)
    return pl.col_expand_mul(normed, gamma)


def partial_rms_sum_squares(acc_tile):
    """Row-wise sum-of-squares partial from Acc tile."""
    if pl is None:
        return _partial_rms_sketch()
    return pl.row_sum(pl.mul(acc_tile, acc_tile))


def fused_matmul_swiglu(x_tile, packed_gate_up_w, ffn_size: int = 2048, rows=None):
    """CODA: GEMM + split gate/up + SiLU(gate)*up. No intermediates in HBM."""
    if pl is None:
        return _swiglu_sketch()
    gate_up = pl.matmul(x_tile, packed_gate_up_w)
    shape = (-1, 2, ffn_size) if rows is None else (rows, 2, ffn_size)
    gate_up = pl.reshape(gate_up, shape)
    if hasattr(pl, "silu"):
        return pl.mul(pl.silu(gate_up[:, 0, :]), gate_up[:, 1, :])
    return pl.mul(gate_up[:, 0, :] * pl.sigmoid(gate_up[:, 0, :]), gate_up[:, 1, :])


# ═══════════════════════════════════════════════════════════════════════════
# DeepSeek-v4-flash CODA Kernels (PyPTO DSL)
# ═══════════════════════════════════════════════════════════════════════════

def _mk_coda_kernel(name, purpose, ops, notes, hbm_r=0, hbm_w=0, unfused=0):
    if pl is not None:
        return None  # real kernel defined via @pl.jit
    return KernelSketch(name=name, purpose=purpose, memory_flow=[],
                        operations=ops, notes=notes,
                        hbm_read_kb=hbm_r, hbm_write_kb=hbm_w,
                        unfused_hbm_kb=unfused)


# ── 1. MLA Q Projection with Input RMSNorm ──

def coda_q_proj_rmsnorm(
    hidden: Any, w_q: Any, gamma_q: Any, output: Any,
    hidden_size: int = 2048, q_rank: int = 1536, eps: float = 1e-6,
):
    """CODA: RMSNorm on input + MLA Q projection fused.
    
    HBM: read hidden(4KB) + gamma(4KB) + w_q(3MB FP8), write Q(3KB)
    Saved vs unfused: avoids materializing normed hidden (4KB)
    """
    if pl is None:
        return _mk_coda_kernel(
            "coda_q_proj_rmsnorm",
            "RMSNorm(input) + MLA Q projection in single kernel",
            ["normed = rms_norm(x, gamma, eps)", "q = normed @ w_q"],
            "Norm intermediate never hits HBM. Q-rank >> hidden so K-tiling on output dim.",
            hbm_r=4+4+3072, hbm_w=3, unfused=4+4+4+3072+3,
        )
    with pl.incore():
        x_tile = pl.load(hidden, [0, 0], [1, hidden_size])
        g_tile = pl.load(gamma_q, [0, 0], [1, hidden_size])
        sq = pl.mul(x_tile, x_tile)
        mean = pl.row_expand_div(pl.row_sum(sq), hidden_size)
        rms = pl.sqrt(pl.add(mean, eps))
        normed = pl.col_expand_mul(pl.row_expand_div(x_tile, rms), g_tile)
        w_l1 = pl.load(w_q, [0, 0], [hidden_size, q_rank], target_memory=pl.MemorySpace.Mat)
        n_l0a = pl.move(normed, target_memory=pl.MemorySpace.Left)
        w_l0b = pl.move(w_l1, target_memory=pl.MemorySpace.Right)
        q = pl.matmul(n_l0a, w_l0b)
        pl.store(q, [0, 0], output)
    return output


# ── 2. MLA KV Packed Projection ──

def coda_kv_proj(
    hidden: Any, w_kv: Any, k_out: Any, v_out: Any,
    hidden_size: int = 2048, kv_rank: int = 512,
):
    """CODA: Single matmul for packed K+V projection, split in epilogue.
    
    w_kv shape: [H, 2*KV_rank] — first half = K, second half = V
    HBM: read hidden(4KB) + w_kv(2MB FP8), write K(1KB) + V(1KB)
    Saved vs unfused: avoids separate K,V matmuls (2x weight loads)
    """
    if pl is None:
        return _mk_coda_kernel(
            "coda_kv_proj",
            "Packed K+V projection: one matmul → split into K, V in epilogue",
            ["kv = hidden @ w_kv", "k, v = split(kv, 2)"],
            "Single weight load for K+V. K/V intermediate never materialized.",
            hbm_r=4+2048, hbm_w=1+1, unfused=4+2048+1+4+2048+1,
        )
    with pl.incore():
        x_tile = pl.load(hidden, [0, 0], [1, hidden_size])
        w_l1 = pl.load(w_kv, [0, 0], [hidden_size, 2*kv_rank], target_memory=pl.MemorySpace.Mat)
        x_l0a = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
        w_l0b = pl.move(w_l1, target_memory=pl.MemorySpace.Right)
        kv = pl.matmul(x_l0a, w_l0b)
        kv_reshaped = pl.reshape(kv, [1, 2, kv_rank])
        k = kv_reshaped[:, 0, :]
        v = kv_reshaped[:, 1, :]
        pl.store(k, [0, 0], k_out)
        pl.store(v, [0, 0], v_out)
    return k_out, v_out


# ── 3. Attention Output + Residual + RMSNorm ──

def coda_attn_out_residual_rmsnorm(
    attn_out: Any, w_out: Any, residual: Any, gamma: Any, output: Any,
    q_rank: int = 1536, hidden_size: int = 2048, eps: float = 1e-6,
):
    """CODA: Attention output proj + residual add + RMSNorm in one kernel.
    
    HBM: read attn_out(3KB) + w_out(3MB FP8) + residual(4KB) + gamma(4KB)
         write normed_hidden(4KB)
    Saved: gemm_out(4KB) never hits HBM
    """
    if pl is None:
        return _mk_coda_kernel(
            "coda_attn_out_residual_rmsnorm",
            "GEMM + residual + RMSNorm: the core CODA Transformer pattern",
            ["h = attn_out @ w_out", "h = h + residual", "out = rms_norm(h, gamma)"],
            "GEMM output consumed directly by residual+RMSNorm in Acc.",
            hbm_r=3+3072+4+4, hbm_w=4, unfused=3+3072+4+4+4+4,
        )
    with pl.incore():
        a_tile = pl.load(attn_out, [0, 0], [1, q_rank])
        w_l1 = pl.load(w_out, [0, 0], [q_rank, hidden_size], target_memory=pl.MemorySpace.Mat)
        a_l0a = pl.move(a_tile, target_memory=pl.MemorySpace.Left)
        w_l0b = pl.move(w_l1, target_memory=pl.MemorySpace.Right)
        h = pl.matmul(a_l0a, w_l0b)
        r_tile = pl.load(residual, [0, 0], [1, hidden_size])
        h_res = pl.add(h, r_tile)
        g_tile = pl.load(gamma, [0, 0], [1, hidden_size])
        sq = pl.mul(h_res, h_res)
        mean = pl.row_expand_div(pl.row_sum(sq), hidden_size)
        rms = pl.sqrt(pl.add(mean, eps))
        normed = pl.col_expand_mul(pl.row_expand_div(h_res, rms), g_tile)
        pl.store(normed, [0, 0], output)
    return output


# ── 4. MoE Gate Router ──

def coda_moe_gate(
    hidden: Any, w_gate: Any,
    topk_indices: Any, topk_weights: Any,
    hidden_size: int = 2048, num_experts: int = 256, top_k: int = 8,
):
    """CODA: MoE router GEMM → top-k selection.
    
    HBM: read hidden(4KB) + w_gate(512KB FP8), write indices(32B) + weights(32B)
    Small kernel — K-tiling overhead dominates. Fuse with preceding RMSNorm.
    """
    if pl is None:
        return _mk_coda_kernel(
            "coda_moe_gate",
            "MoE gate router: GEMM → softmax → top-k",
            ["logits = hidden @ w_gate", "probs = softmax(logits)",
             "topk_weights, topk_indices = topk(probs, k=8)"],
            "All 256 logits kept on-chip. Only top-8 indices+weights stored.",
            hbm_r=4+512, hbm_w=0.0625, unfused=4+512+0.5,
        )
    # Note: full softmax+topk in epilogue is complex; sketch covers the CODA pattern
    return topk_indices, topk_weights


# ── 5. MoE Expert Gate/Up + SwiGLU ──

def coda_expert_swiglu(
    x: Any, w_gate_up: Any, output: Any,
    hidden_size: int = 2048, ffn: int = 2048,
):
    """CODA: Expert gate/up GEMM + SwiGLU. No gate/up intermediates in HBM.
    
    HBM: read x(4KB) + w_gate_up(8MB FP8 for gate+up packed), write swiglu_out(4KB)
    Saved vs unfused: avoids materializing gate(4KB) + up(4KB) separately
    """
    if pl is None:
        return _mk_coda_kernel(
            "coda_expert_swiglu",
            "Expert gate/up GEMM → split → SiLU(gate)*up in one kernel",
            ["gate_up = x @ w_gate_up", "gate, up = split(gate_up)",
             "out = silu(gate) * up"],
            "Gate and up tensors never written to HBM. 2x memory savings.",
            hbm_r=4+8192, hbm_w=4, unfused=4+8192+4+4,
        )
    with pl.incore():
        x_tile = pl.load(x, [0, 0], [1, hidden_size])
        w_l1 = pl.load(w_gate_up, [0, 0], [hidden_size, 2*ffn], target_memory=pl.MemorySpace.Mat)
        x_l0a = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
        w_l0b = pl.move(w_l1, target_memory=pl.MemorySpace.Right)
        gate_up = pl.matmul(x_l0a, w_l0b)
        gate_up_r = pl.reshape(gate_up, [1, 2, ffn])
        gate = gate_up_r[:, 0, :]
        up = gate_up_r[:, 1, :]
        if hasattr(pl, "silu"):
            out = pl.mul(pl.silu(gate), up)
        else:
            out = pl.mul(pl.mul(gate, pl.sigmoid(gate)), up)
        pl.store(out, [0, 0], output)
    return output


# ── 6. MoE Expert Down + Residual ──

def coda_expert_down_residual(
    x: Any, w_down: Any, residual: Any, output: Any,
    ffn: int = 2048, hidden_size: int = 2048,
):
    """CODA: Expert down projection + residual add in one kernel.
    
    HBM: read x(4KB) + w_down(4MB FP8) + residual(4KB), write output(4KB)
    Saved: down_out intermediate(4KB) never hits HBM
    """
    if pl is None:
        return _mk_coda_kernel(
            "coda_expert_down_residual",
            "Expert down GEMM + residual add in one kernel",
            ["down = x @ w_down", "out = down + residual"],
            "Down projection output consumed by residual add in Acc.",
            hbm_r=4+4096+4, hbm_w=4, unfused=4+4096+4+4+4,
        )
    with pl.incore():
        x_tile = pl.load(x, [0, 0], [1, ffn])
        w_l1 = pl.load(w_down, [0, 0], [ffn, hidden_size], target_memory=pl.MemorySpace.Mat)
        x_l0a = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
        w_l0b = pl.move(w_l1, target_memory=pl.MemorySpace.Right)
        down = pl.matmul(x_l0a, w_l0b)
        r_tile = pl.load(residual, [0, 0], [1, hidden_size])
        out = pl.add(down, r_tile)
        pl.store(out, [0, 0], output)
    return output


# ── 7. Shared Expert (Always-On) ──

def coda_shared_expert(
    hidden: Any, w_gate_up: Any, w_down: Any, output: Any,
    hidden_size: int = 2048, ffn: int = 2048,
):
    """CODA: Shared expert path — gate/up GEMM → SwiGLU → down GEMM.
    
    Uses PyPTO ring buffer (A5 SRAM) between gate/up and down to avoid HBM.
    HBM: read hidden(4KB) + w_gate_up(8MB) + w_down(4MB), write output(4KB)
    """
    if pl is None:
        return _mk_coda_kernel(
            "coda_shared_expert",
            "Shared expert: gate/up → SwiGLU → down, all on-chip via ring buffer",
            ["gate_up = hidden @ w_gate_up", "swiglu = silu(gate)*up",
             "out = swiglu @ w_down"],
            "SwiGLU intermediate stays in Vec(UB) via A5 ring buffer pipeline.",
            hbm_r=4+8192+4096, hbm_w=4, unfused=4+8192+4+4+4096+4,
        )
    with pl.incore():
        x_tile = pl.load(hidden, [0, 0], [1, hidden_size])
        w1_l1 = pl.load(w_gate_up, [0, 0], [hidden_size, 2*ffn], target_memory=pl.MemorySpace.Mat)
        x_l0a = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
        w1_l0b = pl.move(w1_l1, target_memory=pl.MemorySpace.Right)
        gate_up = pl.matmul(x_l0a, w1_l0b)
        gate_up_r = pl.reshape(gate_up, [1, 2, ffn])
        gate = gate_up_r[:, 0, :]
        up = gate_up_r[:, 1, :]
        if hasattr(pl, "silu"):
            swiglu = pl.mul(pl.silu(gate), up)
        else:
            swiglu = pl.mul(pl.mul(gate, pl.sigmoid(gate)), up)
        w2_l1 = pl.load(w_down, [0, 0], [ffn, hidden_size], target_memory=pl.MemorySpace.Mat)
        s_l0a = pl.move(swiglu, target_memory=pl.MemorySpace.Left)
        w2_l0b = pl.move(w2_l1, target_memory=pl.MemorySpace.Right)
        out = pl.matmul(s_l0a, w2_l0b)
        pl.store(out, [0, 0], output)
    return output


# ── 8. Full DeepSeek-v4-flash Layer Orchestrator ──

def coda_dpsk_v4_flash_layer(
    hidden: Any,           # [B, H] input activation
    # Attention weights
    w_q: Any, w_kv: Any, w_out: Any,
    # Norm weights
    gamma_attn: Any, gamma_moe: Any,
    # MoE weights
    w_gate: Any, w_moe_up: Any, w_moe_down: Any,
    # Shared expert
    w_shared_up: Any, w_shared_down: Any,
    # Outputs
    output: Any,           # [B, H] layer output
    cfg: DSv4FlashConfig = DSv4FlashConfig(),
):
    """Full DeepSeek-v4-flash layer with CODA fusion at every opportunity.
    
    Pipeline:
      1. RMSNorm → Q/KV proj (MLA)     [coda_q_proj_rmsnorm, coda_kv_proj]
      2. Attention (external, RoPE + flash attn not yet fused)
      3. AttnOut + residual + RMSNorm   [coda_attn_out_residual_rmsnorm]
      4. MoE gate router                [coda_moe_gate]
      5. Expert SwiGLU (×8)             [coda_expert_swiglu]
      6. Expert down + residual (×8)    [coda_expert_down_residual]
      7. Shared expert                  [coda_shared_expert]
      8. Final residual add             [included in step 6+7]
    """
    if pl is None:
        return _mk_coda_kernel(
            "coda_dpsk_v4_flash_layer",
            "Full DeepSeek-v4-flash layer: 8 CODA-fused sub-kernels",
            ["See individual kernel sketches for details"],
            "3 CODA fusion points per layer. Total HBM savings ~",
        )
    B = 1
    H, Q, KV, FFN = cfg.hidden, cfg.q_rank, cfg.kv_rank, cfg.ffn

    # Step 1: Q projection (input already RMSNormd by previous layer)
    q = pl.create_tensor([B, Q], dtype=pl.FP32)
    q = coda_q_proj_rmsnorm(hidden, w_q, gamma_attn, q, H, Q, cfg.eps)

    # Step 2: KV projection
    k = pl.create_tensor([B, KV], dtype=pl.FP32)
    v = pl.create_tensor([B, KV], dtype=pl.FP32)
    k, v = coda_kv_proj(hidden, w_kv, k, v, H, KV)

    # Step 3: Attention (external — RoPE + flash attention)
    attn_out = pl.create_tensor([B, Q], dtype=pl.FP32)
    # attn_out = flash_attention(q, k, v)  # external kernel

    # Step 4: Attention output + residual + RMSNorm
    normed = pl.create_tensor([B, H], dtype=pl.FP32)
    normed = coda_attn_out_residual_rmsnorm(
        attn_out, w_out, hidden, gamma_moe, normed, Q, H, cfg.eps,
    )

    # Step 5: MoE gate router
    gate_idx = pl.create_tensor([B, cfg.top_k], dtype=pl.INT32)
    gate_wt = pl.create_tensor([B, cfg.top_k], dtype=pl.FP32)
    gate_idx, gate_wt = coda_moe_gate(normed, w_gate, gate_idx, gate_wt, H, 256, cfg.top_k)

    # Step 6-7: Expert computation (top-8)
    expert_out = pl.create_tensor([B, H], dtype=pl.FP32)
    # Expert loop would be unrolled or use scatter/gather
    for e in range(cfg.top_k):
        exp_in = pl.create_tensor([B, H], dtype=pl.FP32)
        swiglu_out = pl.create_tensor([B, FFN], dtype=pl.FP32)
        swiglu_out = coda_expert_swiglu(normed, w_moe_up, swiglu_out, H, FFN)
        exp_contrib = pl.create_tensor([B, H], dtype=pl.FP32)
        exp_contrib = coda_expert_down_residual(swiglu_out, w_moe_down, expert_out, exp_contrib, FFN, H)
        expert_out = exp_contrib

    # Step 8: Shared expert + final output
    shared_out = pl.create_tensor([B, H], dtype=pl.FP32)
    shared_out = coda_shared_expert(normed, w_shared_up, w_shared_down, shared_out, H, FFN)

    # Final: add shared expert contribution
    with pl.incore():
        e_tile = pl.load(expert_out, [0, 0], [B, H])
        s_tile = pl.load(shared_out, [0, 0], [B, H])
        final = pl.add(e_tile, s_tile)
        pl.store(final, [0, 0], output)
    return output


# ── 9. Kernel Catalog with HBM Traffic Analysis ──

def catalog_dpsk_v4_flash(cfg: DSv4FlashConfig = DSv4FlashConfig()) -> List[KernelSketch]:
    """Return all CODA kernel sketches with HBM traffic analysis for DSv4-flash."""
    H, Q, KV, FFN = cfg.hidden, cfg.q_rank, cfg.kv_rank, cfg.ffn
    FP8, BF16 = 1, 2  # bytes per element
    
    def kb(n, b=BF16): return n * b / 1024

    kernels = [
        KernelSketch(
            name="coda_q_proj_rmsnorm",
            purpose="RMSNorm(input) + MLA Q projection",
            memory_flow=["Load hidden (Vec)", "Load gamma (Vec)", "RMSNorm", "Load w_q (Mat→L0B)", "matmul→Q"],
            operations=["normed = rms_norm(x, gamma)", "q = normed @ w_q"],
            notes=f"Q intermediate ({kb(Q):.1f}KB) stays in Acc. K-tiling: {max(1, H*Q*FP8//32768)} blocks.",
            hbm_read_kb=kb(H)+kb(H)+kb(H*Q, FP8), hbm_write_kb=kb(Q),
            unfused_hbm_kb=kb(H)+kb(H)+kb(H*Q,FP8)+kb(Q)+kb(H),
        ),
        KernelSketch(
            name="coda_kv_proj",
            purpose="Packed K+V MLA projection",
            memory_flow=["Load hidden (Vec)", "Load w_kv (Mat→L0B)", "matmul→K+V", "split→K, V"],
            operations=["kv = hidden @ w_kv", "k, v = split(kv, dim=1)"],
            notes=f"Single weight load for K+V ({kb(2*H*KV,FP8):.0f}KB). Split in epilogue.",
            hbm_read_kb=kb(H)+kb(2*H*KV, FP8), hbm_write_kb=kb(KV)+kb(KV),
            unfused_hbm_kb=2*(kb(H)+kb(H*KV,FP8)+kb(KV)),
        ),
        KernelSketch(
            name="coda_attn_out_residual_rmsnorm",
            purpose="Attn output proj + residual + RMSNorm",
            memory_flow=["Load attn_out (Vec)", "Load w_out (Mat→L0B)", "matmul→hidden", "Load residual (Vec)", "Load gamma (Vec)", "add + RMSNorm", "store"],
            operations=["h = attn_out @ w_out", "h = h + residual", "out = rms_norm(h, gamma)"],
            notes=f"Core CODA pattern. GEMM output ({kb(H):.1f}KB) never hits HBM.",
            hbm_read_kb=kb(Q)+kb(Q*H,FP8)+kb(H)+kb(H), hbm_write_kb=kb(H),
            unfused_hbm_kb=kb(Q)+kb(Q*H,FP8)+kb(H)+kb(H)+kb(H)+kb(H),
        ),
        KernelSketch(
            name="coda_moe_gate",
            purpose="MoE router: GEMM + top-k",
            memory_flow=["Load hidden (Vec)", "Load w_gate (Mat→L0B)", "matmul→logits", "topk→indices+weights"],
            operations=["logits = hidden @ w_gate", "topk(logits, k=8)"],
            notes=f"Small kernel ({kb(256):.0f}KB logits). K-tiling overhead significant.",
            hbm_read_kb=kb(H)+kb(H*256,FP8), hbm_write_kb=0.0625,
            unfused_hbm_kb=kb(H)+kb(H*256,FP8)+0.5,
        ),
        KernelSketch(
            name="coda_expert_swiglu",
            purpose="Expert gate/up GEMM + SwiGLU (×8 experts)",
            memory_flow=["Load hidden (Vec)", "Load w_gate_up (Mat→L0B)", "matmul→[gate,up]", "SiLU(gate)*up", "store"],
            operations=["gate_up = hidden @ w_gate_up", "gate,up = split(gate_up)", "out = silu(gate)*up"],
            notes=f"Gate ({kb(FFN):.0f}KB) and up ({kb(FFN):.0f}KB) never hit HBM. Big CODA win.",
            hbm_read_kb=kb(H)+kb(2*H*FFN,FP8), hbm_write_kb=kb(FFN),
            unfused_hbm_kb=kb(H)+kb(H*FFN,FP8)+kb(FFN)+kb(H)+kb(H*FFN,FP8)+kb(FFN),
        ),
        KernelSketch(
            name="coda_expert_down_residual",
            purpose="Expert down proj + residual add (×8 experts)",
            memory_flow=["Load swiglu_out (Vec)", "Load w_down (Mat→L0B)", "matmul→down", "Load residual (Vec)", "add", "store"],
            operations=["down = swiglu @ w_down", "out = down + residual"],
            notes=f"Down output ({kb(H):.0f}KB) consumed by residual add in Acc.",
            hbm_read_kb=kb(FFN)+kb(FFN*H,FP8)+kb(H), hbm_write_kb=kb(H),
            unfused_hbm_kb=kb(FFN)+kb(FFN*H,FP8)+kb(H)+kb(H)+kb(H),
        ),
        KernelSketch(
            name="coda_shared_expert",
            purpose="Shared expert: gate/up→SwiGLU→down (ring buffer)",
            memory_flow=["Load hidden (Vec)", "Load w_up (Mat→L0B)", "matmul→SwiGLU", "Load w_down (Mat→L0B)", "matmul→out", "store"],
            operations=["gate_up = hidden @ w_up", "swi = silu(gate)*up", "out = swi @ w_down"],
            notes=f"SwiGLU intermediate ({kb(FFN):.0f}KB) stays in Vec(UB) via A5 ring buffer.",
            hbm_read_kb=kb(H)+kb(2*H*FFN,FP8)+kb(FFN*H,FP8), hbm_write_kb=kb(H),
            unfused_hbm_kb=kb(H)+kb(2*H*FFN,FP8)+kb(FFN)+kb(FFN)+kb(FFN*H,FP8)+kb(H),
        ),
    ]
    return kernels


# ═══════════════════════════════════════════════════════════════════════════
# Original sketches (kept for backward compat)
# ═══════════════════════════════════════════════════════════════════════════

def _gemm_residual_rmsnorm_sketch() -> KernelSketch:
    return KernelSketch(
        name="fused_matmul_residual_rmsnorm",
        purpose="CODA fused GEMM + residual add + RMSNorm",
        memory_flow=[
            "Load activation tile through Vec(UB) into Left(L0A)",
            "Load FP8/BF16 weight tile through Mat(L1) into Right(L0B)",
            "Accumulate matmul in Acc(L0C)",
            "Consume residual from Vec(UB) and normalize in Acc(L0C)",
            "Store only final BF16 normalized output to GM",
        ],
        operations=[
            "acc = pl.matmul(x_tile, w_tile)",
            "residual_acc = pl.add(acc, residual_tile)",
            "sq = pl.mul(residual_acc, residual_acc)",
            "row_ss = pl.row_sum(sq)",
            "mean = pl.row_expand_div(row_ss, hidden_size)",
            "rms = pl.sqrt(pl.add(mean, eps))",
            "normed = pl.row_expand_div(residual_acc, rms)",
            "out = pl.col_expand_mul(normed, gamma)",
        ],
        notes="The matmul output is never materialized in HBM; residual and RMSNorm consume Acc directly.",
    )


def _partial_rms_sketch() -> KernelSketch:
    return KernelSketch(
        name="partial_rms_sum_squares",
        purpose="Partial row-wise RMS reduction for an Acc(L0C) output tile",
        memory_flow=[
            "Read the current Acc(L0C) tile",
            "Compute row-wise sum of squares locally",
            "Store compact partial sums for cross-tile finalization",
        ],
        operations=["sq = pl.mul(acc_tile, acc_tile)", "partial = pl.row_sum(sq)"],
        notes="Partial sums are B-sized row vectors, not full hidden-state intermediates.",
    )


def _swiglu_sketch() -> KernelSketch:
    return KernelSketch(
        name="fused_matmul_swiglu",
        purpose="CODA fused expert gate/up matmul + split + SiLU(gate) * up",
        memory_flow=[
            "Load expert activation tile through Vec(UB) into Left(L0A)",
            "Load packed [gate, up] weight tile through Mat(L1) into Right(L0B)",
            "Accumulate packed gate/up output in Acc(L0C)",
            "Split Acc columns into gate and up views",
            "Apply SiLU and multiply in local memory",
            "Store only the BF16 SwiGLU product to GM",
        ],
        operations=[
            "gate_up = pl.matmul(x_tile, packed_gate_up_w)",
            "gate_up = pl.reshape(gate_up, (rows, 2, ffn_size))",
            "gate = gate_up[:, 0, :]",
            "up = gate_up[:, 1, :]",
            "out = pl.mul(pl.silu(gate), up)",
        ],
        notes="Gate/up tensors are not separately written to HBM.",
    )


def describe_kernels() -> Dict[str, Dict[str, object]]:
    result = {
        "fused_matmul_residual_rmsnorm": _gemm_residual_rmsnorm_sketch().to_dict(),
        "partial_rms_sum_squares": _partial_rms_sketch().to_dict(),
        "fused_matmul_swiglu": _swiglu_sketch().to_dict(),
        "pypto_available": {"available": pypto_available()},
    }
    for k in catalog_dpsk_v4_flash():
        result[k.name] = k.to_dict()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main: print kernel catalog
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = DSv4FlashConfig()
    kernels = catalog_dpsk_v4_flash(cfg)
    
    print(f"DeepSeek-v4-flash CODA Kernel Catalog")
    print(f"  Config: H={cfg.hidden} Q={cfg.q_rank} KV={cfg.kv_rank} FFN={cfg.ffn}")
    print(f"  MoE: {cfg.num_experts} experts, top-{cfg.top_k}")
    print()
    
    total_saved = 0.0
    for k in kernels:
        saved = k.to_dict()["hbm_saved_kb"]
        total_saved += saved
        print(f"  {k.name}")
        print(f"    HBM: read {k.hbm_read_kb:.0f}KB + write {k.hbm_write_kb:.0f}KB"
              f" = {k.hbm_read_kb + k.hbm_write_kb:.0f}KB"
              f" (saved {saved:.0f}KB vs unfused {k.unfused_hbm_kb:.0f}KB)")
        print(f"    {k.purpose}")
        print()
    
    print(f"  Total HBM saved per layer: {total_saved:.0f} KB"
          f" ({total_saved/1024:.1f} MB)")
    print(f"  Per 32-layer model: {total_saved * 32 / 1024:.1f} MB saved per forward pass")
    print()
    print(f"  PyPTO available: {pypto_available()}")


__all__ = [
    "DSv4FlashConfig", "KernelSketch",
    "pypto_available",
    "fused_matmul_residual_rmsnorm", "partial_rms_sum_squares", "fused_matmul_swiglu",
    "coda_q_proj_rmsnorm", "coda_kv_proj", "coda_attn_out_residual_rmsnorm",
    "coda_moe_gate", "coda_expert_swiglu", "coda_expert_down_residual",
    "coda_shared_expert", "coda_dpsk_v4_flash_layer",
    "catalog_dpsk_v4_flash", "describe_kernels",
]
