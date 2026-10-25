import sys

import pytest

from amagi_throughput_lab.power import PowerSampler, integrate, parse_line


def test_parse_line():
    assert parse_line("0, 291.72, 66, 22204") == (0, 291.72, 66.0, 22204.0)
    assert parse_line("1, 100.00 W, [N/A], 10 MiB") == (1, 100.0, None, 10.0)
    assert parse_line("0, [N/A], 66, 1") is None
    assert parse_line("garbage") is None


def test_integrate_constant_power():
    samples = [(t, 0, 200.0) for t in range(0, 11)]
    joules, avg, n = integrate(samples, 2.0, 7.0)
    assert joules == pytest.approx(1000.0) and avg == pytest.approx(200.0) and n == 6


def test_integrate_ramp_and_held_edges():
    samples = [(0.0, 0, 100.0), (10.0, 0, 300.0)]
    joules, _, _ = integrate(samples, 0.0, 10.0)
    assert joules == pytest.approx(2000.0)
    # A window between two samples takes the nearest sample on each side as its edges.
    joules, _, n = integrate(samples, 4.0, 6.0)
    assert n == 0 and joules == pytest.approx(400.0)


def test_integrate_sums_gpus_and_filters():
    samples = [(t, g, 100.0 * (g + 1)) for t in range(5) for g in (0, 1)]
    assert integrate(samples, 0, 4)[0] == pytest.approx(1200.0)
    assert integrate(samples, 0, 4, gpus={1})[0] == pytest.approx(800.0)


def test_sampler_window_with_fed_lines():
    s = PowerSampler()
    for t, line in [(0, "0, 250, 60, 1000"), (1, "0, 250, 70, 1500"), (2, "0, 250, 65, 1200")]:
        s.feed(line, t=t)
    s.feed("nvidia-smi says no", t=3)
    win = s.window(0, 2)
    assert win["energy_j"] == pytest.approx(500.0)
    assert win["max_temp_c"] == 70 and win["peak_mem_mib"] == 1500
    assert s.errors == ["nvidia-smi says no"]
    assert s.latest_temp() == 70


def test_sampler_reads_a_real_subprocess(tmp_path):
    # Stands in for nvidia-smi: prints the same CSV shape and ignores its flags.
    script = tmp_path / "fake_smi.py"
    script.write_text(
        "import time\n"
        "for i in range(100):\n"
        "    print('0, 123.5, 61, 900', flush=True)\n"
        "    time.sleep(0.02)\n", encoding="utf-8")
    sampler = PowerSampler([sys.executable, str(script)])
    sampler.start()
    try:
        assert sampler.wait_for_samples(10)
    finally:
        sampler.stop()
    assert sampler.samples[0][2] == 123.5


def test_missing_nvidia_smi_is_a_clear_error():
    with pytest.raises(RuntimeError, match="--no-power"):
        PowerSampler(["no-such-nvidia-smi-binary"]).start()
