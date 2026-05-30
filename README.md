# CODA on Ascend 950 for DeepSeek-v4-flash

Refined hardware model + CODA kernel implementation in PyPTO DSL.

## Structure
- `hw_model.py` — Enhanced Ascend 950 model with K-tiling cost, multi-batch, multi-dtype
- `baseline_pt.py` — Unfused PyTorch reference kernels
- `coda_pypto.py` — CODA-style fused kernels in PyPTO DSL
- `analyze.py` — Run analysis: roofline, CODA savings, PTO IR inspection
