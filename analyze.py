from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

from coda_pypto import describe_kernels
from hw_model import Ascend950, DeepSeekV4FlashConfig


RESULTS_PATH = Path("results.json")


def human_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def human_flops(value: float) -> str:
    units = ["FLOP", "KFLOP", "MFLOP", "GFLOP", "TFLOP"]
    size = float(value)
    for unit in units:
        if abs(size) < 1000.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1000.0
    return f"{size:.2f} TFLOP"


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    rows_as_text = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows_as_text)) for idx in range(len(headers))
    ]
    header_line = "  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    rule = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    body = [
        "  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) for row in rows_as_text
    ]
    return "\n".join([header_line, rule, *body])


def print_hardware(hw: Ascend950) -> None:
    print("Ascend 950 hardware summary")
    print(f"- Dies: {len(hw.soc.dies)}")
    print(f"- Clusters: {hw.total_clusters} total, 18 per die")
    print(f"- Cube cores: {hw.total_cube_cores}; Vector cores: {hw.total_vector_cores}")
    print("- Per Cube: Mat(L1)=512KB, Left(L0A)=64KB, Right(L0B)=64KB, Acc(L0C)=256KB, Bias=4KB")
    print("- Per Vector: Vec(UB)=248KB")
    print("- A5 ring buffer: consumer-local SRAM for Cube<->Vector, no GM round-trip")
    print(f"- Peak BF16 compute: {hw.peak_bf16_tflops:.1f} TFLOP/s")
    print(f"- HBM bandwidth: {hw.hbm_bandwidth_gb_s:.0f} GB/s")
    print(f"- Ridge point: {hw.ridge_flop_per_byte:.1f} FLOP/byte")
    print(
        f"- K-tiling overhead: k_blocks * ({hw.launch_overhead_us:.0f}us launch + "
        f"{hw.pipeline_fill_us:.0f}us/stage * {hw.pipeline_stages} stage)"
    )


def print_model(cfg: DeepSeekV4FlashConfig) -> None:
    print("\nDeepSeek-v4-flash model summary")
    print(f"- Hidden size: {cfg.hidden_size}")
    print(f"- Layers: {cfg.layers}")
    print(f"- MLA ranks: Q={cfg.q_rank}, KV={cfg.kv_rank}")
    print(f"- MoE: {cfg.num_experts} experts, top-{cfg.top_k}")
    print(f"- FFN size: {cfg.ffn_size}")
    print(f"- Weight mode: {cfg.weight_mode.upper()} params, BF16 compute/activations")
    print(f"- RMSNorm eps: {cfg.rmsnorm_eps}")


def operator_rows(hw: Ascend950, cfg: DeepSeekV4FlashConfig, batch: int, weight_mode: str):
    catalog = hw.operator_catalog(cfg)
    rows = []
    results = {}
    for name, op in catalog.items():
        without = hw.roofline(op, batch=batch, weight_mode=weight_mode, include_k_tiling=False)
        with_overhead = hw.roofline(op, batch=batch, weight_mode=weight_mode, include_k_tiling=True)
        rows.append(
            [
                name,
                human_flops(with_overhead.flops),
                human_bytes(with_overhead.traffic.total_bytes),
                f"{with_overhead.arithmetic_intensity_flop_per_byte:.1f}",
                with_overhead.k_blocks,
                f"{without.time_us:.3f}",
                f"{with_overhead.k_tiling_overhead_us:.3f}",
                f"{with_overhead.time_us:.3f}",
                with_overhead.bottleneck,
            ]
        )
        results[name] = {
            "without_k_tiling": without.to_dict(),
            "with_k_tiling": with_overhead.to_dict(),
        }
    return rows, results


def print_operator_catalog(hw: Ascend950, cfg: DeepSeekV4FlashConfig) -> dict:
    print("\nPer-operator roofline, B=1, FP8 weights")
    rows, fp8_results = operator_rows(hw, cfg, batch=1, weight_mode="fp8")
    print(
        table(
            [
                "Operator",
                "FLOPs",
                "HBM traffic",
                "AI",
                "Kblk",
                "NoK us",
                "K us",
                "Total us",
                "Bottleneck",
            ],
            rows,
        )
    )

    print("\nPer-operator roofline, B=1, BF16 weights")
    bf16_rows, bf16_results = operator_rows(hw, cfg, batch=1, weight_mode="bf16")
    print(
        table(
            [
                "Operator",
                "FLOPs",
                "HBM traffic",
                "AI",
                "Kblk",
                "NoK us",
                "K us",
                "Total us",
                "Bottleneck",
            ],
            bf16_rows,
        )
    )

    return {"fp8": fp8_results, "bf16": bf16_results}


def print_fusion_savings(hw: Ascend950, cfg: DeepSeekV4FlashConfig, batch: int) -> dict:
    savings = hw.coda_fusion_savings(batch=batch, cfg=cfg)
    rows = []
    for name, entry in savings.items():
        rows.append(
            [
                name,
                human_bytes(entry["unfused_bytes"]),
                human_bytes(entry["fused_bytes"]),
                human_bytes(entry["saved_bytes"]),
                f"{100.0 * entry['saved_fraction']:.1f}%",
            ]
        )
    print(f"\nCODA fused vs unfused HBM traffic, B={batch}")
    print(table(["Fusion", "Unfused", "Fused", "Saved", "Saved %"], rows))
    return savings


def print_multi_batch(hw: Ascend950, cfg: DeepSeekV4FlashConfig) -> List[dict]:
    print("\nMulti-batch model throughput with FP8 weight reuse")
    rows = []
    results = []
    for item in hw.multi_batch_analysis((1, 4, 16, 64, 256), cfg=cfg, weight_mode="fp8"):
        rows.append(
            [
                item.batch,
                f"{item.tokens_per_second:,.1f}",
                f"{100.0 * item.bandwidth_utilization_fraction:.2f}%",
                human_bytes(item.hbm_traffic_per_token_bytes),
                human_bytes(item.total_traffic_bytes),
            ]
        )
        results.append(item.to_dict(hw.hbm_bytes_per_s))
    print(table(["B", "tok/s", "BW util", "HBM/token", "HBM/pass"], rows))
    print("- Weight reuse model: parameters are read once per batch; BF16 activations scale with B.")
    return results


def print_layer_breakdown(hw: Ascend950, cfg: DeepSeekV4FlashConfig) -> dict:
    decode = hw.full_model_throughput(batch=1, cfg=cfg, weight_mode="fp8")
    batched_decode = hw.full_model_throughput(batch=256, cfg=cfg, weight_mode="fp8")
    layer_b1 = decode.layer
    layer_b256 = batched_decode.layer

    print("\nFull 32-layer model breakdown")
    rows = [
        [
            "Decode B=1",
            human_bytes(decode.total_traffic_bytes),
            human_bytes(layer_b1.coda_saved_bytes * cfg.layers),
            f"{decode.tokens_per_second:,.1f}",
            layer_b1.bottleneck,
        ],
        [
            "Batched decode proxy B=256",
            human_bytes(batched_decode.total_traffic_bytes),
            human_bytes(layer_b256.coda_saved_bytes * cfg.layers),
            f"{batched_decode.tokens_per_second:,.1f}",
            layer_b256.bottleneck,
        ],
    ]
    print(table(["Mode", "HBM/pass", "CODA saved/pass", "tok/s", "Layer bottleneck"], rows))

    return {
        "decode_b1": decode.to_dict(hw.hbm_bytes_per_s),
        "batched_decode_b256": batched_decode.to_dict(hw.hbm_bytes_per_s),
    }


def print_prefill_decode_comparison(
    hw: Ascend950,
    cfg: DeepSeekV4FlashConfig,
    batch: int = 1,
    prompt_length: int = 256,
) -> dict:
    comparison = hw.prefill_decode_comparison(
        batch=batch,
        prompt_length=prompt_length,
        cfg=cfg,
        weight_mode="fp8",
    )

    print(f"\nFair prefill/decode comparison, B={batch}, prompt/generated tokens={prompt_length}")
    rows = []
    labels = {
        "prefill": "Prefill one prompt",
        "decode": "Decode serial steps",
    }
    for key in ("prefill", "decode"):
        item = comparison[key]
        rows.append(
            [
                labels[key],
                item.total_tokens,
                item.attention_pairs,
                item.projection_passes,
                f"{item.linear_layer_time_us:.1f}",
                f"{item.attention_layer_time_us:.1f}",
                f"{item.total_time_us / 1000.0:.1f}",
                f"{item.tokens_per_second:,.1f}",
                human_bytes(item.hbm_traffic_per_token_bytes),
            ]
        )
    print(
        table(
            [
                "Mode",
                "Tokens",
                "Attn pairs",
                "Linear passes",
                "Linear us/layer",
                "Attn us/layer",
                "Model ms",
                "tok/s",
                "HBM/token",
            ],
            rows,
        )
    )
    print("- Fairness rule: both rows produce the same token count and causal attention pairs.")
    print("- Difference measured here is launch/weight-reuse amortization, not different work.")

    return {key: item.to_dict(hw.hbm_bytes_per_s) for key, item in comparison.items()}


def main() -> None:
    hw = Ascend950()
    cfg = DeepSeekV4FlashConfig()

    print_hardware(hw)
    print_model(cfg)
    operator_results = print_operator_catalog(hw, cfg)
    fusion_b1 = print_fusion_savings(hw, cfg, batch=1)
    fusion_b256 = print_fusion_savings(hw, cfg, batch=256)
    multi_batch = print_multi_batch(hw, cfg)
    full_model = print_layer_breakdown(hw, cfg)
    prefill_decode_fair = print_prefill_decode_comparison(hw, cfg)

    results = {
        "hardware": hw.hardware_summary(),
        "model": cfg.to_dict(),
        "operators": operator_results,
        "fusion_savings": {"B1": fusion_b1, "B256": fusion_b256},
        "multi_batch": multi_batch,
        "full_model": full_model,
        "prefill_decode_fair": prefill_decode_fair,
        "coda_kernels": describe_kernels(),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved structured results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
