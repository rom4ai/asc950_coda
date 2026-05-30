import math

from hw_model import Ascend950, DeepSeekV4FlashConfig


def test_ascend950_topology_and_memory_sizes():
    hw = Ascend950()

    assert len(hw.soc.dies) == 2
    assert hw.total_clusters == 36
    assert hw.total_cube_cores == 36
    assert hw.total_vector_cores == 72

    cluster = hw.soc.dies[0].clusters[0]
    assert cluster.cube.memory["Mat(L1)"].size_kb == 512
    assert cluster.cube.memory["Left(L0A)"].size_kb == 64
    assert cluster.cube.memory["Right(L0B)"].size_kb == 64
    assert cluster.cube.memory["Acc(L0C)"].size_kb == 256
    assert cluster.cube.memory["Bias"].size_kb == 4
    assert cluster.vectors[0].memory["Vec(UB)"].size_kb == 248


def test_k_tiling_uses_32kb_l0b_rule_and_power_of_two_alignment():
    hw = Ascend950()

    assert hw.k_tiling_blocks(k_dim=2048, n_tile=16, weight_mode="fp8") == 1
    assert hw.k_tiling_blocks(k_dim=2048, n_tile=16, weight_mode="bf16") == 2
    assert hw.k_tiling_blocks(k_dim=2048, n_tile=80, weight_mode="fp8") == 8


def test_operator_roofline_includes_optional_k_tiling_overhead():
    hw = Ascend950()
    cfg = DeepSeekV4FlashConfig()
    q_proj = hw.operator_catalog(cfg)["Q_proj"]

    no_overhead = hw.roofline(q_proj, batch=1, weight_mode="fp8", include_k_tiling=False)
    with_overhead = hw.roofline(q_proj, batch=1, weight_mode="fp8", include_k_tiling=True)

    assert with_overhead.k_blocks == 1
    assert math.isclose(with_overhead.k_tiling_overhead_us, 7.0)
    assert with_overhead.time_us > no_overhead.time_us
    assert with_overhead.bottleneck in {"compute", "memory", "k_tiling_overhead"}


def test_weight_reuse_reduces_hbm_traffic_per_token_as_batch_grows():
    hw = Ascend950()
    cfg = DeepSeekV4FlashConfig()
    q_proj = hw.operator_catalog(cfg)["Q_proj"]

    b1 = hw.estimate_operator(q_proj, batch=1, weight_mode="fp8")
    b256 = hw.estimate_operator(q_proj, batch=256, weight_mode="fp8")

    assert b256.traffic.total_bytes > b1.traffic.total_bytes
    assert b256.traffic.total_bytes / 256 < b1.traffic.total_bytes
    assert b256.traffic.weight_bytes == b1.traffic.weight_bytes


def test_coda_fusion_savings_are_positive_and_scale_with_batch():
    hw = Ascend950()
    cfg = DeepSeekV4FlashConfig()

    b1 = hw.coda_fusion_savings(batch=1, cfg=cfg)
    b256 = hw.coda_fusion_savings(batch=256, cfg=cfg)

    assert b1["gemm_residual_rmsnorm"]["saved_bytes"] > 0
    assert b1["gemm_swiglu"]["saved_bytes"] > b1["gemm_residual_rmsnorm"]["saved_bytes"]
    assert b256["total_per_layer"]["saved_bytes"] == b1["total_per_layer"]["saved_bytes"] * 256
