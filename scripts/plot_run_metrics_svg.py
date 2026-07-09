import argparse
import csv
import html
from pathlib import Path


def to_float(value):
    if value in (None, ""):
        return None
    return float(value)


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        item = {"type": row.get("type", ""), "step": int(float(row["step"]))}
        for key in (
            "tokens_seen",
            "train_loss",
            "val_loss",
            "val_perplexity",
            "tokens_per_sec",
            "total_tokens_per_sec",
            "gpu_memory_gb",
        ):
            item[key] = to_float(row.get(key))
        parsed.append(item)
    return parsed


def get_points(rows, key, row_type=None):
    points = []
    for row in rows:
        if row_type is not None and row["type"] != row_type:
            continue
        value = row.get(key)
        tokens = row.get("tokens_seen")
        if value is not None and tokens is not None:
            points.append((tokens / 1_000_000, value))
    return points


def scale(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def polyline(points, x, y, width, height):
    if not points:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else max(1.0, abs(y_max) * 0.08)
    y_min -= y_pad
    y_max += y_pad
    coords = []
    for px, py in points:
        sx = scale(px, x_min, x_max, x + 48, x + width - 16)
        sy = scale(py, y_min, y_max, y + height - 34, y + 18)
        coords.append(f"{sx:.1f},{sy:.1f}")
    return " ".join(coords)


def panel(rows, key, title, ylabel, x, y, width, height, row_type=None):
    points = get_points(rows, key, row_type=row_type)
    values = [p[1] for p in points]
    tokens = [p[0] for p in points]
    escaped_title = html.escape(title)
    escaped_ylabel = html.escape(ylabel)
    pieces = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#ffffff" stroke="#d0d7de"/>',
        f'<text x="{x + 14}" y="{y + 24}" font-size="15" font-weight="700">{escaped_title}</text>',
        f'<text x="{x + 14}" y="{y + height - 10}" font-size="11" fill="#57606a">tokens seen, millions</text>',
        f'<text x="{x + 12}" y="{y + 46}" font-size="11" fill="#57606a">{escaped_ylabel}</text>',
    ]
    for i in range(5):
        gy = y + 38 + i * ((height - 80) / 4)
        pieces.append(f'<line x1="{x + 48}" y1="{gy:.1f}" x2="{x + width - 16}" y2="{gy:.1f}" stroke="#eaeef2"/>')
    if points:
        line = polyline(points, x, y, width, height)
        pieces.append(f'<polyline points="{line}" fill="none" stroke="#0969da" stroke-width="2.2"/>')
        pieces.append(
            f'<text x="{x + width - 16}" y="{y + 24}" text-anchor="end" font-size="11" fill="#57606a">'
            f'{min(tokens):.1f}-{max(tokens):.1f}M, {min(values):.3g}-{max(values):.3g}</text>'
        )
    else:
        pieces.append(f'<text x="{x + width / 2}" y="{y + height / 2}" text-anchor="middle" fill="#57606a">no data</text>')
    return "\n".join(pieces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv")
    parser.add_argument("--output", default="plots/run_metrics.svg")
    parser.add_argument("--title", default="forge-LLM training metrics")
    args = parser.parse_args()

    rows = load_rows(args.metrics_csv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1380, 760
    panels = [
        ("train_loss", "Training loss", "loss", "train"),
        ("val_loss", "Validation loss", "loss", None),
        ("val_perplexity", "Validation perplexity", "perplexity", None),
        ("tokens_per_sec", "Logged throughput", "tokens/sec", "train"),
        ("total_tokens_per_sec", "Average throughput", "tokens/sec", "train"),
        ("gpu_memory_gb", "CUDA memory", "GB", "train"),
    ]
    panel_w, panel_h = 430, 280
    gap_x, gap_y = 24, 34
    start_x, start_y = 24, 88

    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f8fa"/>',
        f'<text x="24" y="42" font-size="26" font-weight="800">{html.escape(args.title)}</text>',
        f'<text x="24" y="66" font-size="13" fill="#57606a">Generated from {html.escape(args.metrics_csv)}</text>',
    ]
    for index, (key, title, ylabel, row_type) in enumerate(panels):
        row = index // 3
        col = index % 3
        x = start_x + col * (panel_w + gap_x)
        y = start_y + row * (panel_h + gap_y)
        content.append(panel(rows, key, title, ylabel, x, y, panel_w, panel_h, row_type=row_type))
    content.append("</svg>")

    output.write_text("\n".join(content), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
