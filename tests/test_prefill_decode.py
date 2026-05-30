from hw_model import BF16_BYTES, Ascend950, DeepSeekV4FlashConfig


def test_prefill_decode_comparison_is_token_and_attention_fair():
    hw = Ascend950()
    cfg = DeepSeekV4FlashConfig()

    comparison = hw.prefill_decode_comparison(batch=1, prompt_length=16, cfg=cfg)
    prefill = comparison["prefill"]
    decode = comparison["decode"]

    expected_pairs = 16 * 17 // 2
    assert prefill.total_tokens == decode.total_tokens == 16
    assert prefill.attention_pairs == decode.attention_pairs == expected_pairs
    assert prefill.projection_passes == 1
    assert decode.projection_passes == 16
    assert decode.linear_layer_time_us > prefill.linear_layer_time_us


def test_decode_workload_accounts_for_existing_context_and_kv_cache_traffic():
    hw = Ascend950()
    cfg = DeepSeekV4FlashConfig()

    decode = hw.decode_workload(batch=2, context_length=8, generated_tokens=3, cfg=cfg)

    expected_pairs = 2 * sum(8 + step + 1 for step in range(3))
    expected_kv_read = expected_pairs * cfg.kv_rank * 2 * BF16_BYTES
    expected_kv_write = 2 * 3 * cfg.kv_rank * 2 * BF16_BYTES

    assert decode.total_tokens == 6
    assert decode.attention_pairs == expected_pairs
    assert decode.attention.kv_cache_read_bytes == expected_kv_read
    assert decode.attention.kv_cache_write_bytes == expected_kv_write


def test_prefill_uses_one_linear_pass_over_all_prompt_tokens():
    hw = Ascend950()
    cfg = DeepSeekV4FlashConfig()

    prefill = hw.prefill_workload(batch=2, prompt_length=8, cfg=cfg)
    equivalent_layer = hw.full_layer_roofline(batch=16, cfg=cfg)

    assert prefill.total_tokens == 16
    assert prefill.projection_passes == 1
    assert prefill.attention_pairs == 2 * 8 * 9 // 2
    assert prefill.linear_layer_time_us == equivalent_layer.time_us
