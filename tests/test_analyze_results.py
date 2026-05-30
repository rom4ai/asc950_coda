import json

import analyze


def test_analyze_results_include_fair_prefill_decode_comparison(tmp_path, monkeypatch, capsys):
    results_path = tmp_path / "results.json"
    monkeypatch.setattr(analyze, "RESULTS_PATH", results_path)

    analyze.main()

    captured = capsys.readouterr()
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert "Fair prefill/decode comparison" in captured.out
    comparison = data["prefill_decode_fair"]
    assert comparison["prefill"]["total_tokens"] == comparison["decode"]["total_tokens"]
    assert comparison["prefill"]["attention_pairs"] == comparison["decode"]["attention_pairs"]
    assert comparison["decode"]["projection_passes"] > comparison["prefill"]["projection_passes"]


def test_analyze_results_cover_recommendation_sweeps(tmp_path, monkeypatch, capsys):
    results_path = tmp_path / "results.json"
    monkeypatch.setattr(analyze, "RESULTS_PATH", results_path)

    analyze.main()

    capsys.readouterr()
    data = json.loads(results_path.read_text(encoding="utf-8"))

    decode_sweep = data["decode_context_sweep"]
    prefill_sweep = data["prefill_length_sweep"]
    persistent = data["persistent_kernel_sensitivity"]

    assert [item["context_length"] for item in decode_sweep] == [0, 128, 512, 2048]
    assert all(item["generated_tokens"] == 1 for item in decode_sweep)
    assert decode_sweep[-1]["attention_pairs"] > decode_sweep[0]["attention_pairs"]
    assert all(item["attention"]["kv_cache_read_bytes"] >= 0 for item in decode_sweep)
    assert all(item["attention"]["kv_cache_write_bytes"] > 0 for item in decode_sweep)

    assert [item["prompt_length"] for item in prefill_sweep] == [64, 128, 256, 512]
    assert prefill_sweep[-1]["attention_pairs"] > prefill_sweep[0]["attention_pairs"]
    assert all(item["projection_passes"] == 1 for item in prefill_sweep)

    assert set(persistent) == {"decode", "prefill"}
    for mode in ("decode", "prefill"):
        assert persistent[mode]["no_k_tiling"]["total_time_us"] < persistent[mode]["baseline"]["total_time_us"]
        assert persistent[mode]["baseline"]["mode"] == mode
        assert persistent[mode]["no_k_tiling"]["mode"] == mode
