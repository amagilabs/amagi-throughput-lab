"""Charts for a sweep: one self-contained HTML page (inline SVG) and one PNG.

Three line charts, each with one y-axis and concurrency on a log2 x-axis:
total tok/s, per-person tok/s (slowest 10%), and tok/J when power was sampled.
One line per server layout, at most 8 (the best by peak throughput); the rest
stay in the table. Colors are a validated colorblind-safe categorical order in
fixed slot order; every line is also direct-labeled and listed in the legend,
and the full numbers are in the table under the charts.
"""

import html
import math

from PIL import Image, ImageDraw, ImageFont

from .recommend import describe, layout_key, layout_name

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9",
               "#e66767"]
SURFACE, INK, MUTED, GRID = "#F6F4F1", "#161616", "#6b6762", "#DDD8D2"
MAX_SERIES = 8

CHARTS = [
    ("throughput_tps", "Total throughput", "tokens/s, all requests together"),
    ("user_tps_p10", "What the slowest 10% of people see", "tokens/s per request, 10th percentile"),
    ("tok_per_j", "Energy efficiency", "tokens per joule (board power from nvidia-smi)"),
]


def short_layout(key):
    inst, par, batch, ubatch = key
    text = f"{inst}x np{par if par is not None else '?'}"
    if batch is not None:
        text += f" b{batch}"
    if ubatch is not None:
        text += f" ub{ubatch}"
    return text


def series_for(rows, metric):
    """[(label, [(concurrency, value), ...])] ordered by each layout's peak throughput."""
    groups = {}
    for r in rows:
        if r.get("status") != "ok" or r.get(metric) in (None, 0):
            continue
        groups.setdefault(layout_key(r), []).append(r)
    peak = {k: max(x.get("throughput_tps") or 0 for x in v) for k, v in groups.items()}
    order = sorted(groups, key=lambda k: (-peak[k], k[0] or 0, k[1] or 0))
    return order, {k: sorted((x["concurrency"], x[metric]) for x in groups[k]) for k in order}


def nice_max(value):
    if value <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * exp:
            return step * exp
    return 10 * exp


class Frame:
    """Shared chart geometry for the SVG and PNG renderers."""

    def __init__(self, xs, ymax, width=760, height=300, left=64, right=110, top=16, bottom=44):
        self.w, self.h = width, height
        self.l, self.r, self.t, self.b = left, right, top, bottom
        self.xs = sorted(set(xs))
        self.lx0 = math.log2(self.xs[0]) if self.xs else 0
        self.lx1 = math.log2(self.xs[-1]) if self.xs else 1
        self.ymax = nice_max(ymax)

    def x(self, c):
        span = (self.lx1 - self.lx0) or 1.0
        return self.l + (math.log2(c) - self.lx0) / span * (self.w - self.l - self.r)

    def y(self, v):
        return self.t + (1 - v / self.ymax) * (self.h - self.t - self.b)

    def yticks(self):
        return [self.ymax * i / 4 for i in range(5)]


def tick_label(v):
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}".rstrip("0").rstrip(".") if v else "0"


def build_charts(rows):
    charts = []
    for metric, title, unit in CHARTS:
        order, data = series_for(rows, metric)
        if not order:
            continue
        shown = order[:MAX_SERIES]
        xs = [c for k in shown for c, _ in data[k]]
        ymax = max(v for k in shown for _, v in data[k])
        charts.append({"metric": metric, "title": title, "unit": unit, "frame": Frame(xs, ymax),
                       "series": [(short_layout(k), SERIES[i], SERIES_DARK[i], data[k])
                                  for i, k in enumerate(shown)],
                       "hidden": len(order) - len(shown)})
    return charts


def label_positions(series, frame, min_gap=13):
    """End-of-line label y positions, nudged apart so they don't overlap.

    Labels are pushed down to clear each other, then pulled back up as a group if
    that would run them past the x-axis.
    """
    ends = sorted(((frame.y(pts[-1][1]), i) for i, (_, _, _, pts) in enumerate(series)))
    ys, last = [], -1e9
    for y, _ in ends:
        last = max(y, last + min_gap)
        ys.append(last)
    limit = frame.h - frame.b
    for k in reversed(range(len(ys))):
        ys[k] = min(ys[k], limit)
        limit = ys[k] - min_gap
    return {i: y for (_, i), y in zip(ends, ys)}


def label_leader(frame, pts, label_y):
    """Points for a short line in the series color from the last point to its label."""
    ex, ey = frame.x(pts[-1][0]), frame.y(pts[-1][1])
    return [(ex + 6, ey), (ex + 12, label_y), (ex + 22, label_y)], ex + 26


# ---------------------------------------------------------------- SVG / HTML
def svg_chart(chart, idx):
    f = chart["frame"]
    out = [f'<svg viewBox="0 0 {f.w} {f.h}" role="img" aria-labelledby="t{idx}" class="chart">',
           f'<title id="t{idx}">{html.escape(chart["title"])}</title>']
    for v in f.yticks():
        y = f.y(v)
        out.append(f'<line x1="{f.l}" x2="{f.w - f.r}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{f.l - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">'
                   f'{tick_label(v)}</text>')
    for c in f.xs:
        out.append(f'<text x="{f.x(c):.1f}" y="{f.h - f.b + 18}" text-anchor="middle" '
                   f'class="tick">{c}</text>')
    out.append(f'<text x="{(f.l + f.w - f.r) / 2:.0f}" y="{f.h - 6}" text-anchor="middle" '
               'class="axis">requests in flight (concurrency)</text>')
    labels = label_positions(chart["series"], f)
    for i, (name, color, _dark, pts) in enumerate(chart["series"]):
        var = f"var(--s{i + 1})"
        path = " ".join(f"{'M' if j == 0 else 'L'}{f.x(c):.1f},{f.y(v):.1f}"
                        for j, (c, v) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{var}" stroke-width="2" '
                   'stroke-linejoin="round" stroke-linecap="round"/>')
        for c, v in pts:
            tip = f"{name} at {c} in flight: {v:,.2f}"
            out.append(f'<g class="pt"><circle cx="{f.x(c):.1f}" cy="{f.y(v):.1f}" r="12" '
                       f'fill="transparent"/><circle cx="{f.x(c):.1f}" cy="{f.y(v):.1f}" r="4" '
                       f'fill="{var}" stroke="var(--surface)" stroke-width="2"/>'
                       f'<title>{html.escape(tip)}</title></g>')
        leader, text_x = label_leader(f, pts, labels[i])
        out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in leader)}" '
                   f'fill="none" stroke="{var}" stroke-width="2"/>')
        out.append(f'<text x="{text_x:.1f}" y="{labels[i] + 4:.1f}" class="lbl">'
                   f'{html.escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)


TABLE_COLS = [("instances", "inst", 0), ("parallel", "np", 0), ("batch", "batch", 0),
              ("ubatch", "ubatch", 0), ("concurrency", "in flight", 0),
              ("throughput_tps", "total tok/s", 1), ("user_tps_p50", "per-person p50", 1),
              ("user_tps_p10", "slowest 10%", 1), ("latency_p95_s", "p95 wait s", 1),
              ("avg_power_w", "avg W", 0), ("tok_per_j", "tok/J", 2), ("max_temp_c", "max C", 0),
              ("errors", "errors", 0), ("status", "status", None)]


def cell(value, digits):
    if value is None or value == "":
        return ""
    if digits is None or isinstance(value, str):
        return html.escape(str(value))
    return f"{value:,.{digits}f}"


def render_html(rows, meta, rec=None, model=None):
    meta = meta or {}
    charts = build_charts(rows)
    gpus = ", ".join(meta.get("gpus") or []) or "GPU not recorded"
    cfg = meta.get("config", {})
    sub = (f"{html.escape(cfg.get('model') or 'model')} on {html.escape(gpus)} | "
           f"{html.escape(cfg.get('backend', ''))} {html.escape(meta.get('server_version', ''))} | "
           f"{cfg.get('n_predict', '')} new tokens per request, prompts: "
           f"{html.escape(str(cfg.get('prompt_mode', '')))}")
    css_vars = "".join(f"--s{i + 1}:{c};" for i, c in enumerate(SERIES))
    css_dark = "".join(f"--s{i + 1}:{c};" for i, c in enumerate(SERIES_DARK))
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>amagi-throughput-lab report</title><style>",
        f":root{{--surface:{SURFACE};--ink:{INK};--muted:{MUTED};--grid:{GRID};--code:#ebe7e2;{css_vars}}}",
        "@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#ffffff;"
        f"--muted:#c3c2b7;--grid:#3a3936;--code:#262524;{css_dark}}}}}",
        "body{margin:0;background:var(--surface);color:var(--ink);"
        "font:15px/1.45 system-ui,-apple-system,'Segoe UI',sans-serif}",
        "main{max-width:900px;margin:0 auto;padding:28px 20px 60px}",
        "h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:30px 0 2px}",
        ".sub,.unit,.foot{color:var(--muted)}.unit{margin:0 0 6px;font-size:13px}",
        ".banner{background:#eda10033;border:1px solid #c98500;padding:8px 12px;border-radius:6px;"
        "margin:14px 0}",
        ".chart{width:100%;height:auto;display:block}",
        ".grid{stroke:var(--grid);stroke-width:1}.tick{fill:var(--muted);font-size:11px}",
        ".axis{fill:var(--muted);font-size:12px}.lbl{fill:var(--ink);font-size:12px}",
        ".pt:hover circle:last-of-type{r:6}",
        ".legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;margin:4px 0 0}",
        ".legend span::before{content:'';display:inline-block;width:14px;height:3px;"
        "border-radius:2px;margin-right:6px;vertical-align:middle;background:var(--c)}",
        "pre{background:var(--code);padding:12px;border-radius:6px;overflow:auto;white-space:pre-wrap;font-size:13px}",
        "table{border-collapse:collapse;font-size:12.5px;width:100%;margin-top:6px}",
        "th,td{padding:4px 6px;border-bottom:1px solid var(--grid);text-align:right}",
        "th{color:var(--muted);font-weight:600}td:last-child,th:last-child{text-align:left}",
        "</style></head><body><main>",
        "<h1>Throughput sweep</h1>",
        f"<p class=\"sub\">{sub}</p>",
    ]
    if meta.get("simulated"):
        parts.append("<p class=\"banner\"><strong>SIMULATED.</strong> These numbers came from the "
                     "built-in fake server, not a real model. Use them to try the tool, not to "
                     "size hardware.</p>")
    if meta.get("stopped_early"):
        parts.append(f"<p class=\"banner\">Sweep stopped early: "
                     f"{html.escape(meta['stopped_early'])}</p>")
    if rec is not None:
        parts.append("<h2>Recommendation</h2>")
        parts.append(f"<pre>{html.escape(describe(rec, meta, model=model))}</pre>")
    for i, ch in enumerate(charts):
        parts.append(f"<h2>{html.escape(ch['title'])}</h2><p class=\"unit\">"
                     f"{html.escape(ch['unit'])}</p>")
        parts.append(svg_chart(ch, i))
        parts.append("<div class=\"legend\">" + "".join(
            f"<span style=\"--c:var(--s{j + 1})\">{html.escape(name)}</span>"
            for j, (name, _c, _d, _p) in enumerate(ch["series"])) + "</div>")
        if ch["hidden"]:
            parts.append(f"<p class=\"unit\">{ch['hidden']} more layout(s) are in the table "
                         "below; the chart shows the 8 with the highest peak throughput.</p>")
    parts.append("<h2>All cells</h2><table><thead><tr>"
                 + "".join(f"<th>{h}</th>" for _k, h, _d in TABLE_COLS) + "</tr></thead><tbody>")
    for r in rows:
        parts.append("<tr>" + "".join(f"<td>{cell(r.get(k), d)}</td>" for k, _h, d in TABLE_COLS)
                     + "</tr>")
    parts.append("</tbody></table>")
    parts.append("<p class=\"foot\">Made with amagi-throughput-lab, based on llama-throughput-lab "
                 "by Alex Ziskind (MIT). Per-person tok/s is a request's generated tokens over "
                 "its full wait, queueing and prefill included.</p></main></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------- PNG
def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without FreeType size support
        return ImageFont.load_default()


def render_png(rows, meta, path, scale=2):
    meta = meta or {}
    charts = build_charts(rows)
    if not charts:
        raise ValueError("no successful cells to chart")
    head = 70
    fw, fh = charts[0]["frame"].w, charts[0]["frame"].h
    block = fh + 30
    img = Image.new("RGB", (fw * scale, (head + block * len(charts) + 16) * scale), SURFACE)
    d = ImageDraw.Draw(img)
    s = scale
    f_title, f_sub, f_tick = _font(20 * s), _font(12 * s), _font(11 * s)
    cfg = meta.get("config", {})
    gpus = ", ".join(meta.get("gpus") or []) or "GPU not recorded"
    d.text((20 * s, 14 * s), "Throughput sweep" + ("  [SIMULATED]" if meta.get("simulated") else ""),
           fill=INK, font=f_title)
    d.text((20 * s, 42 * s), f"{cfg.get('model') or 'model'} on {gpus} | {cfg.get('backend', '')} | "
           f"{cfg.get('n_predict', '')} new tokens/request | amagi-throughput-lab",
           fill=MUTED, font=f_sub)
    for n, ch in enumerate(charts):
        f = ch["frame"]
        oy = head + n * block
        d.text((20 * s, (oy + 4) * s), f"{ch['title']}  ({ch['unit']})", fill=INK, font=f_sub)
        oy += 10

        def P(x, y):
            return (x * s, (y + oy) * s)

        for v in f.yticks():
            y = f.y(v)
            d.line([P(f.l, y), P(f.w - f.r, y)], fill=GRID, width=s)
            d.text(P(f.l - 8, y - 6), tick_label(v), fill=MUTED, font=f_tick, anchor="ra")
        for c in f.xs:
            d.text(P(f.x(c), f.h - f.b + 8), str(c), fill=MUTED, font=f_tick, anchor="ma")
        d.text(P((f.l + f.w - f.r) / 2, f.h - 18), "requests in flight (concurrency)",
               fill=MUTED, font=f_tick, anchor="ma")
        labels = label_positions(ch["series"], f)
        for i, (name, color, _dark, pts) in enumerate(ch["series"]):
            xy = [P(f.x(c), f.y(v)) for c, v in pts]
            if len(xy) > 1:
                d.line(xy, fill=color, width=2 * s, joint="curve")
            for px, py in xy:
                r = 4 * s
                d.ellipse([px - r - 2 * s, py - r - 2 * s, px + r + 2 * s, py + r + 2 * s],
                          fill=SURFACE)
                d.ellipse([px - r, py - r, px + r, py + r], fill=color)
            leader, text_x = label_leader(f, pts, labels[i])
            d.line([P(x, y) for x, y in leader], fill=color, width=2 * s)
            d.text(P(text_x, labels[i] - 6), name, fill=INK, font=f_tick)
    img.save(path, optimize=True)
    return path
