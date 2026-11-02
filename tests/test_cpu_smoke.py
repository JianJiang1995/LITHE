import cpu_smoke


def test_cpu_smoke(tmp_path):
    r = cpu_smoke.run(steps=200, out=str(tmp_path), verbose=False)
    assert r["eval_mse_after"] < 0.7 * r["eval_mse_before"]
    assert r["dense_vs_lithe_max_abs_diff"] < 1e-5
    assert r["indices_roundtrip_identical"]
