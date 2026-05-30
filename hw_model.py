from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Dict, Iterable, List, Mapping, Optional


BF16_BYTES = 2
FP8_BYTES = 1
L0B_K_TILE_BYTES = 32 * 1024


def _round_up_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _require_weight_mode(weight_mode: str) -> str:
    normalized = weight_mode.lower()
    if normalized not in {"fp8", "bf16"}:
        raise ValueError(f"weight_mode must be 'fp8' or 'bf16', got {weight_mode!r}")
    return normalized


def weight_bytes_per_element(weight_mode: str) -> int:
    return FP8_BYTES if _require_weight_mode(weight_mode) == "fp8" else BF16_BYTES


@dataclass(frozen=True)
class MemoryRegion:
    name: str
    size_kb: int

    @property
    def size_bytes(self) -> int:
        return self.size_kb * 1024


@dataclass(frozen=True)
class CubeCore:
    die_id: int
    cluster_id: int
    core_id: int
    memory: Mapping[str, MemoryRegion] = field(default_factory=dict)

    @classmethod
    def create(cls, die_id: int, cluster_id: int, core_id: int) -> "CubeCore":
        memory = {
            "Mat(L1)": MemoryRegion("Mat(L1)", 512),
            "Left(L0A)": MemoryRegion("Left(L0A)", 64),
            "Right(L0B)": MemoryRegion("Right(L0B)", 64),
            "Acc(L0C)": MemoryRegion("Acc(L0C)", 256),
            "Bias": MemoryRegion("Bias", 4),
        }
        return cls(die_id=die_id, cluster_id=cluster_id, core_id=core_id, memory=memory)


@dataclass(frozen=True)
class VectorCore:
    die_id: int
    cluster_id: int
    core_id: int
    memory: Mapping[str, MemoryRegion] = field(default_factory=dict)

    @classmethod
    def create(cls, die_id: int, cluster_id: int, core_id: int) -> "VectorCore":
        return cls(
            die_id=die_id,
            cluster_id=cluster_id,
            core_id=core_id,
            memory={"Vec(UB)": MemoryRegion("Vec(UB)", 248)},
        )


@dataclass(frozen=True)
class Cluster:
    die_id: int
    cluster_id: int
    cube: CubeCore
    vectors: List[VectorCore]
    ring_buffer: str = "A5 consumer-local SRAM for Cube<->Vector handoff"

    @classmethod
    def create(cls, die_id: int, cluster_id: int) -> "Cluster":
        cube = CubeCore.create(die_id=die_id, cluster_id=cluster_id, core_id=cluster_id)
        vector_base = cluster_id * 2
        vectors = [
            VectorCore.create(die_id=die_id, cluster_id=cluster_id, core_id=vector_base),
            VectorCore.create(die_id=die_id, cluster_id=cluster_id, core_id=vector_base + 1),
        ]
        return cls(die_id=die_id, cluster_id=cluster_id, cube=cube, vectors=vectors)


@dataclass(frozen=True)
class Die:
    die_id: int
    clusters: List[Cluster]

    @classmethod
    def create(cls, die_id: int, clusters_per_die: int) -> "Die":
        cluster_offset = die_id * clusters_per_die
        clusters = [Cluster.create(die_id, cluster_offset + idx) for idx in range(clusters_per_die)]
        return cls(die_id=die_id, clusters=clusters)


@dataclass(frozen=True)
class SoCTopology:
    dies: List[Die]

    @classmethod
    def create(cls, dies: int = 2, clusters_per_die: int = 18) -> "SoCTopology":
        return cls(dies=[Die.create(die_id, clusters_per_die) for die_id in range(dies)])

    @property
    def clusters(self) -> List[Cluster]:
        return [cluster for die in self.dies for cluster in die.clusters]

    def to_dict(self) -> Dict[str, object]:
        return {
            "dies": [
                {
                    "die_id": die.die_id,
                    "clusters": [
                        {
                            "cluster_id": cluster.cluster_id,
                            "cube_core_id": cluster.cube.core_id,
                            "vector_core_ids": [vector.core_id for vector in cluster.vectors],
                            "cube_memory_kb": {
                                name: region.size_kb for name, region in cluster.cube.memory.items()
                            },
                            "vector_memory_kb": {
                                name: region.size_kb
                                for name, region in cluster.vectors[0].memory.items()
                            },
                            "ring_buffer": cluster.ring_buffer,
                        }
                        for cluster in die.clusters
                    ],
                }
                for die in self.dies
            ]
        }


@dataclass(frozen=True)
class DeepSeekV4FlashConfig:
    hidden_size: int = 2048
    layers: int = 32
    q_rank: int = 1536
    kv_rank: int = 512
    num_experts: int = 256
    top_k: int = 8
    ffn_size: int = 2048
    weight_mode: str = "fp8"
    compute_dtype: str = "bf16"
    activation_dtype: str = "bf16"
    rmsnorm_eps: float = 1e-6

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    kind: str
    flops_per_token: int
    activation_read_elements_per_token: int
    activation_write_elements_per_token: int
    weight_elements: int = 0
    weight_precision: str = "mode"
    k_dim: Optional[int] = None
    n_dim: Optional[int] = None
    n_tile: int = 16
    description: str = ""

    @property
    def uses_k_tiling(self) -> bool:
        return self.k_dim is not None and self.n_dim is not None and self.kind in {
            "gemm",
            "moe_gemm",
            "composite_gemm",
        }

    def bytes_per_weight(self, weight_mode: str) -> int:
        if self.weight_precision == "bf16":
            return BF16_BYTES
        if self.weight_precision == "none":
            return 0
        return weight_bytes_per_element(weight_mode)


@dataclass(frozen=True)
class HBMTraffic:
    weight_bytes: int
    activation_read_bytes: int
    activation_write_bytes: int
    intermediate_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return (
            self.weight_bytes
            + self.activation_read_bytes
            + self.activation_write_bytes
            + self.intermediate_bytes
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "weight_bytes": self.weight_bytes,
            "activation_read_bytes": self.activation_read_bytes,
            "activation_write_bytes": self.activation_write_bytes,
            "intermediate_bytes": self.intermediate_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class OperatorEstimate:
    name: str
    batch: int
    weight_mode: str
    flops: int
    traffic: HBMTraffic
    arithmetic_intensity_flop_per_byte: float
    compute_time_us: float
    bandwidth_time_us: float
    k_blocks: int
    k_tiling_overhead_us: float
    time_us: float
    bottleneck: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "batch": self.batch,
            "weight_mode": self.weight_mode,
            "flops": self.flops,
            "traffic": self.traffic.to_dict(),
            "arithmetic_intensity_flop_per_byte": self.arithmetic_intensity_flop_per_byte,
            "compute_time_us": self.compute_time_us,
            "bandwidth_time_us": self.bandwidth_time_us,
            "k_blocks": self.k_blocks,
            "k_tiling_overhead_us": self.k_tiling_overhead_us,
            "time_us": self.time_us,
            "bottleneck": self.bottleneck,
        }


@dataclass(frozen=True)
class LayerEstimate:
    batch: int
    weight_mode: str
    operators: List[OperatorEstimate]
    total_flops: int
    unfused_traffic_bytes: int
    coda_saved_bytes: int
    traffic_after_coda_bytes: int
    compute_time_us: float
    bandwidth_time_us: float
    k_tiling_overhead_us: float
    time_us: float
    bottleneck: str

    @property
    def bandwidth_utilization(self) -> float:
        if self.time_us == 0:
            return 0.0
        achieved_bps = self.traffic_after_coda_bytes / (self.time_us * 1e-6)
        return achieved_bps

    def to_dict(self, peak_hbm_bytes_per_s: float) -> Dict[str, object]:
        return {
            "batch": self.batch,
            "weight_mode": self.weight_mode,
            "operators": [estimate.to_dict() for estimate in self.operators],
            "total_flops": self.total_flops,
            "unfused_traffic_bytes": self.unfused_traffic_bytes,
            "coda_saved_bytes": self.coda_saved_bytes,
            "traffic_after_coda_bytes": self.traffic_after_coda_bytes,
            "compute_time_us": self.compute_time_us,
            "bandwidth_time_us": self.bandwidth_time_us,
            "k_tiling_overhead_us": self.k_tiling_overhead_us,
            "time_us": self.time_us,
            "bottleneck": self.bottleneck,
            "bandwidth_utilization_fraction": self.bandwidth_utilization / peak_hbm_bytes_per_s,
        }


@dataclass(frozen=True)
class ModelThroughput:
    batch: int
    layers: int
    weight_mode: str
    total_time_us: float
    tokens_per_second: float
    total_traffic_bytes: int
    hbm_traffic_per_token_bytes: float
    bandwidth_utilization_fraction: float
    layer: LayerEstimate

    def to_dict(self, peak_hbm_bytes_per_s: float) -> Dict[str, object]:
        return {
            "batch": self.batch,
            "layers": self.layers,
            "weight_mode": self.weight_mode,
            "total_time_us": self.total_time_us,
            "tokens_per_second": self.tokens_per_second,
            "total_traffic_bytes": self.total_traffic_bytes,
            "hbm_traffic_per_token_bytes": self.hbm_traffic_per_token_bytes,
            "bandwidth_utilization_fraction": self.bandwidth_utilization_fraction,
            "layer": self.layer.to_dict(peak_hbm_bytes_per_s),
        }


@dataclass
class Ascend950:
    soc: SoCTopology = field(default_factory=SoCTopology.create)
    peak_bf16_tflops: float = 354.0
    hbm_bandwidth_gb_s: float = 2400.0
    launch_overhead_us: float = 5.0
    pipeline_fill_us: float = 2.0
    pipeline_stages: int = 1

    @property
    def total_clusters(self) -> int:
        return len(self.soc.clusters)

    @property
    def total_cube_cores(self) -> int:
        return self.total_clusters

    @property
    def total_vector_cores(self) -> int:
        return self.total_clusters * 2

    @property
    def peak_flops_per_s(self) -> float:
        return self.peak_bf16_tflops * 1e12

    @property
    def hbm_bytes_per_s(self) -> float:
        return self.hbm_bandwidth_gb_s * 1e9

    @property
    def ridge_flop_per_byte(self) -> float:
        return self.peak_flops_per_s / self.hbm_bytes_per_s

    def hardware_summary(self) -> Dict[str, object]:
        return {
            "dies": len(self.soc.dies),
            "clusters_per_die": len(self.soc.dies[0].clusters),
            "total_clusters": self.total_clusters,
            "total_cube_cores": self.total_cube_cores,
            "total_vector_cores": self.total_vector_cores,
            "peak_bf16_tflops": self.peak_bf16_tflops,
            "hbm_bandwidth_gb_s": self.hbm_bandwidth_gb_s,
            "ridge_flop_per_byte": self.ridge_flop_per_byte,
            "launch_overhead_us": self.launch_overhead_us,
            "pipeline_fill_us_per_stage": self.pipeline_fill_us,
            "pipeline_stages": self.pipeline_stages,
            "topology": self.soc.to_dict(),
        }

    def operator_catalog(self, cfg: Optional[DeepSeekV4FlashConfig] = None) -> Dict[str, OperatorSpec]:
        cfg = cfg or DeepSeekV4FlashConfig()
        h = cfg.hidden_size
        q = cfg.q_rank
        kv = cfg.kv_rank
        ffn = cfg.ffn_size
        top_k = cfg.top_k
        experts = cfg.num_experts

        return {
            "Q_proj": OperatorSpec(
                name="Q_proj",
                kind="gemm",
                k_dim=h,
                n_dim=q,
                flops_per_token=2 * h * q,
                activation_read_elements_per_token=h,
                activation_write_elements_per_token=q,
                weight_elements=h * q,
                description="MLA query projection H -> Q-rank",
            ),
            "KV_proj": OperatorSpec(
                name="KV_proj",
                kind="gemm",
                k_dim=h,
                n_dim=kv,
                flops_per_token=2 * h * kv,
                activation_read_elements_per_token=h,
                activation_write_elements_per_token=kv,
                weight_elements=h * kv,
                description="MLA compressed key/value projection H -> KV-rank",
            ),
            "AttnOut": OperatorSpec(
                name="AttnOut",
                kind="gemm",
                k_dim=q,
                n_dim=h,
                flops_per_token=2 * q * h,
                activation_read_elements_per_token=q,
                activation_write_elements_per_token=h,
                weight_elements=q * h,
                description="Attention output projection Q-rank -> H",
            ),
            "MoE_gate": OperatorSpec(
                name="MoE_gate",
                kind="gemm",
                k_dim=h,
                n_dim=experts,
                flops_per_token=2 * h * experts,
                activation_read_elements_per_token=h,
                activation_write_elements_per_token=experts,
                weight_elements=h * experts,
                description="Router projection H -> 256 expert logits",
            ),
            "MoE_gate_up": OperatorSpec(
                name="MoE_gate_up",
                kind="moe_gemm",
                k_dim=h,
                n_dim=2 * ffn,
                flops_per_token=top_k * 2 * h * (2 * ffn),
                activation_read_elements_per_token=top_k * h,
                activation_write_elements_per_token=top_k * (2 * ffn),
                weight_elements=top_k * h * (2 * ffn),
                description="Top-8 expert fused gate/up projection H -> 2*FFN",
            ),
            "MoE_down": OperatorSpec(
                name="MoE_down",
                kind="moe_gemm",
                k_dim=ffn,
                n_dim=h,
                flops_per_token=top_k * 2 * ffn * h,
                activation_read_elements_per_token=top_k * ffn,
                activation_write_elements_per_token=h,
                weight_elements=top_k * ffn * h,
                description="Top-8 expert down projection FFN -> H",
            ),
            "RMSNorm": OperatorSpec(
                name="RMSNorm",
                kind="norm",
                flops_per_token=5 * h,
                activation_read_elements_per_token=h,
                activation_write_elements_per_token=h,
                weight_elements=h,
                weight_precision="bf16",
                description="BF16 RMSNorm with epsilon 1e-6",
            ),
            "Residual": OperatorSpec(
                name="Residual",
                kind="elementwise",
                flops_per_token=h,
                activation_read_elements_per_token=2 * h,
                activation_write_elements_per_token=h,
                weight_elements=0,
                weight_precision="none",
                description="Residual add on hidden state",
            ),
            "SharedExpert": OperatorSpec(
                name="SharedExpert",
                kind="composite_gemm",
                k_dim=h,
                n_dim=2 * ffn,
                flops_per_token=(2 * h * (2 * ffn)) + (2 * ffn * h),
                activation_read_elements_per_token=h,
                activation_write_elements_per_token=h,
                weight_elements=(h * (2 * ffn)) + (ffn * h),
                description="Always-on shared expert gate/up plus down path",
            ),
        }

    def layer_operator_names(self) -> List[str]:
        return [
            "RMSNorm",
            "Q_proj",
            "KV_proj",
            "AttnOut",
            "Residual",
            "RMSNorm",
            "MoE_gate",
            "MoE_gate_up",
            "MoE_down",
            "SharedExpert",
            "Residual",
        ]

    def k_tiling_blocks(self, k_dim: int, n_tile: int = 16, weight_mode: str = "fp8") -> int:
        bytes_needed = k_dim * n_tile * weight_bytes_per_element(weight_mode)
        raw_blocks = ceil(bytes_needed / L0B_K_TILE_BYTES)
        return _round_up_power_of_two(max(1, raw_blocks))

    def k_tiling_overhead_us(self, op: OperatorSpec, weight_mode: str = "fp8") -> float:
        if not op.uses_k_tiling:
            return 0.0
        blocks = self.k_tiling_blocks(op.k_dim or 0, op.n_tile, weight_mode)
        per_block = self.launch_overhead_us + self.pipeline_fill_us * self.pipeline_stages
        return blocks * per_block

    def estimate_operator(
        self,
        op: OperatorSpec,
        batch: int,
        weight_mode: str = "fp8",
        include_k_tiling: bool = True,
    ) -> OperatorEstimate:
        weight_mode = _require_weight_mode(weight_mode)
        flops = op.flops_per_token * batch
        traffic = HBMTraffic(
            weight_bytes=op.weight_elements * op.bytes_per_weight(weight_mode),
            activation_read_bytes=op.activation_read_elements_per_token * batch * BF16_BYTES,
            activation_write_bytes=op.activation_write_elements_per_token * batch * BF16_BYTES,
        )
        compute_time_us = flops / self.peak_flops_per_s * 1e6
        bandwidth_time_us = traffic.total_bytes / self.hbm_bytes_per_s * 1e6
        k_blocks = (
            self.k_tiling_blocks(op.k_dim or 0, op.n_tile, weight_mode) if op.uses_k_tiling else 0
        )
        overhead_us = self.k_tiling_overhead_us(op, weight_mode) if include_k_tiling else 0.0
        base_time_us = max(compute_time_us, bandwidth_time_us)
        time_us = base_time_us + overhead_us
        arithmetic_intensity = flops / traffic.total_bytes if traffic.total_bytes else float("inf")
        bottleneck = "compute" if compute_time_us >= bandwidth_time_us else "memory"
        if overhead_us > base_time_us:
            bottleneck = "k_tiling_overhead"

        return OperatorEstimate(
            name=op.name,
            batch=batch,
            weight_mode=weight_mode,
            flops=flops,
            traffic=traffic,
            arithmetic_intensity_flop_per_byte=arithmetic_intensity,
            compute_time_us=compute_time_us,
            bandwidth_time_us=bandwidth_time_us,
            k_blocks=k_blocks,
            k_tiling_overhead_us=overhead_us,
            time_us=time_us,
            bottleneck=bottleneck,
        )

    def roofline(
        self,
        op: OperatorSpec,
        batch: int,
        weight_mode: str = "fp8",
        include_k_tiling: bool = True,
    ) -> OperatorEstimate:
        return self.estimate_operator(op, batch, weight_mode, include_k_tiling)

    def coda_fusion_savings(
        self,
        batch: int,
        cfg: Optional[DeepSeekV4FlashConfig] = None,
    ) -> Dict[str, Dict[str, float]]:
        cfg = cfg or DeepSeekV4FlashConfig()
        hidden_tensor_bytes = batch * cfg.hidden_size * BF16_BYTES
        expert_tensor_bytes = batch * cfg.top_k * cfg.ffn_size * BF16_BYTES

        # Conservative accounting requested by the spec: count one GEMM output
        # tensor materialization as a write plus a later read from HBM.
        gemm_residual_saved = 2 * hidden_tensor_bytes
        swiglu_saved = 2 * (expert_tensor_bytes + expert_tensor_bytes)

        gemm_fused_bytes = hidden_tensor_bytes + cfg.hidden_size * BF16_BYTES + hidden_tensor_bytes
        gemm_unfused_bytes = gemm_fused_bytes + gemm_residual_saved
        swiglu_fused_bytes = expert_tensor_bytes
        swiglu_unfused_bytes = swiglu_fused_bytes + swiglu_saved
        total_saved = 2 * gemm_residual_saved + swiglu_saved
        total_unfused = 2 * gemm_unfused_bytes + swiglu_unfused_bytes

        return {
            "gemm_residual_rmsnorm": {
                "saved_bytes": gemm_residual_saved,
                "unfused_bytes": gemm_unfused_bytes,
                "fused_bytes": gemm_fused_bytes,
                "saved_fraction": gemm_residual_saved / gemm_unfused_bytes,
            },
            "gemm_swiglu": {
                "saved_bytes": swiglu_saved,
                "unfused_bytes": swiglu_unfused_bytes,
                "fused_bytes": swiglu_fused_bytes,
                "saved_fraction": swiglu_saved / swiglu_unfused_bytes,
            },
            "total_per_layer": {
                "saved_bytes": total_saved,
                "unfused_bytes": total_unfused,
                "fused_bytes": total_unfused - total_saved,
                "saved_fraction": total_saved / total_unfused,
            },
        }

    def full_layer_roofline(
        self,
        batch: int,
        cfg: Optional[DeepSeekV4FlashConfig] = None,
        weight_mode: str = "fp8",
        apply_coda_savings: bool = True,
        include_k_tiling: bool = True,
    ) -> LayerEstimate:
        cfg = cfg or DeepSeekV4FlashConfig()
        catalog = self.operator_catalog(cfg)
        operators = [
            self.estimate_operator(catalog[name], batch, weight_mode, include_k_tiling)
            for name in self.layer_operator_names()
        ]
        total_flops = sum(estimate.flops for estimate in operators)
        unfused_traffic = sum(estimate.traffic.total_bytes for estimate in operators)
        saved_bytes = (
            int(self.coda_fusion_savings(batch, cfg)["total_per_layer"]["saved_bytes"])
            if apply_coda_savings
            else 0
        )
        traffic_after_coda = max(0, unfused_traffic - saved_bytes)
        compute_time_us = total_flops / self.peak_flops_per_s * 1e6
        bandwidth_time_us = traffic_after_coda / self.hbm_bytes_per_s * 1e6
        overhead_us = sum(estimate.k_tiling_overhead_us for estimate in operators)
        base_time_us = max(compute_time_us, bandwidth_time_us)
        time_us = base_time_us + overhead_us
        bottleneck = "compute" if compute_time_us >= bandwidth_time_us else "memory"
        if overhead_us > base_time_us:
            bottleneck = "k_tiling_overhead"

        return LayerEstimate(
            batch=batch,
            weight_mode=weight_mode,
            operators=operators,
            total_flops=total_flops,
            unfused_traffic_bytes=unfused_traffic,
            coda_saved_bytes=saved_bytes,
            traffic_after_coda_bytes=traffic_after_coda,
            compute_time_us=compute_time_us,
            bandwidth_time_us=bandwidth_time_us,
            k_tiling_overhead_us=overhead_us,
            time_us=time_us,
            bottleneck=bottleneck,
        )

    def full_model_throughput(
        self,
        batch: int,
        cfg: Optional[DeepSeekV4FlashConfig] = None,
        weight_mode: str = "fp8",
        apply_coda_savings: bool = True,
        include_k_tiling: bool = True,
    ) -> ModelThroughput:
        cfg = cfg or DeepSeekV4FlashConfig()
        layer = self.full_layer_roofline(
            batch=batch,
            cfg=cfg,
            weight_mode=weight_mode,
            apply_coda_savings=apply_coda_savings,
            include_k_tiling=include_k_tiling,
        )
        total_time_us = layer.time_us * cfg.layers
        total_traffic = layer.traffic_after_coda_bytes * cfg.layers
        seconds = total_time_us * 1e-6
        tok_s = batch / seconds if seconds else 0.0
        achieved_bps = total_traffic / seconds if seconds else 0.0

        return ModelThroughput(
            batch=batch,
            layers=cfg.layers,
            weight_mode=weight_mode,
            total_time_us=total_time_us,
            tokens_per_second=tok_s,
            total_traffic_bytes=total_traffic,
            hbm_traffic_per_token_bytes=total_traffic / batch,
            bandwidth_utilization_fraction=achieved_bps / self.hbm_bytes_per_s,
            layer=layer,
        )

    def multi_batch_analysis(
        self,
        batches: Iterable[int] = (1, 4, 16, 64, 256),
        cfg: Optional[DeepSeekV4FlashConfig] = None,
        weight_mode: str = "fp8",
    ) -> List[ModelThroughput]:
        cfg = cfg or DeepSeekV4FlashConfig()
        return [
            self.full_model_throughput(batch, cfg=cfg, weight_mode=weight_mode)
            for batch in batches
        ]
