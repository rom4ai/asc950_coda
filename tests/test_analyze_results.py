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
