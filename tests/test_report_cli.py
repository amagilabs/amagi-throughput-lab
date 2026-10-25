import shutil

from PIL import Image

from amagi_throughput_lab.cli import build_parser, int_list, main, opt_int_list
from amagi_throughput_lab.recommend import recommend
from amagi_throughput_lab.report import MAX_SERIES, build_charts, render_html, render_png
from amagi_throughput_lab.sweep import load_results


def test_fixture_loads_with_types(fixtures):
    rows, meta = load_results(fixtures / "sample")
    assert rows[0]["instances"] == 1 and isinstance(rows[0]["throughput_tps"], float)
    assert meta["config"]["backend"] == "llamacpp"


def test_html_has_charts_table_legend_and_recommendation(fixtures):
    rows, meta = load_results(fixtures / "sample")
    page = render_html(rows, meta, recommend(rows, users=10, min_user_tps=10))
    assert page.count("<svg") == 3  # throughput, slowest-10%, tok/J
    assert "<table>" in page and "class=\"legend\"" in page
    assert "Run it with:" in page and "prefers-color-scheme: dark" in page
    assert "SIMULATED" not in page  # fixture meta is not marked simulated
    assert "Alex Ziskind" in page


def test_simulated_banner(fixtures):
    rows, meta = load_results(fixtures / "sample")
    assert "SIMULATED" in render_html(rows, dict(meta, simulated=True))


def test_charts_cap_series_and_skip_missing_metrics(fixtures):
    rows, _ = load_results(fixtures / "sample")
    many = [dict(r, instances=i) for i in range(1, 12) for r in rows]
    charts = build_charts(many)
    assert all(len(c["series"]) <= MAX_SERIES for c in charts)
    assert charts[0]["hidden"] > 0
    no_power = [dict(r, tok_per_j=None) for r in rows]
    assert [c["metric"] for c in build_charts(no_power)] == ["throughput_tps", "user_tps_p10"]


def test_png(fixtures, tmp_path):
    rows, meta = load_results(fixtures / "sample")
    path = render_png(rows, meta, tmp_path / "r.png")
    with Image.open(path) as img:
        assert img.width == 1520 and img.height > 1000


def test_report_command_writes_into_results_dir(fixtures, tmp_path):
    src = tmp_path / "sample"
    shutil.copytree(fixtures / "sample", src)
    assert main(["report", str(src)]) == 0
    assert (src / "report.html").exists() and (src / "report.png").exists()


def test_recommend_command_on_fixture(fixtures, capsys):
    assert main(["recommend", str(fixtures / "sample"), "--users", "10", "--nginx"]) == 0
    out = capsys.readouterr().out
    assert "For 10 people" in out and "llama-server" in out


def test_cli_errors(tmp_path, capsys):
    assert main(["recommend", str(tmp_path)]) == 3
    assert "no sweep.csv" in capsys.readouterr().err
    assert main([]) == 2
    assert main(["sweep", "--no-power"]) == 2  # no --model and no --attach


def test_list_parsing():
    assert int_list("1,2, 4") == [1, 2, 4]
    args = build_parser().parse_args(["sweep", "-m", "x", "--batch", "default"])
    assert args.batch == [None]


def test_prompts_command(capsys):
    assert main(["prompts", "-n", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3 and len(set(lines)) == 3


def test_batch_lists_can_mix_the_server_default_with_numbers():
    assert opt_int_list("default") == [None]
    assert opt_int_list("default,512,2048") == [None, 512, 2048]
    assert opt_int_list("512,1024") == [512, 1024]
