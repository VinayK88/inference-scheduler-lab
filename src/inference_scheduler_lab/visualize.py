from __future__ import annotations

from html import escape


def render_policy_frontier(report: dict) -> str:
    """Return a dependency-free SVG of throughput versus tail latency."""
    results = report["results"]
    width, height = 920, 520
    left, top, plot_width, plot_height = 100, 60, 720, 350
    throughputs = [item["throughput_output_tokens_per_second"] for item in results]
    latencies = [item["p95_latency_ms"] for item in results]
    min_x, max_x = min(throughputs) * 0.90, max(throughputs) * 1.08
    min_y, max_y = 0.0, max(latencies) * 1.12

    def x(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_width

    def y(value: float) -> float:
        return top + plot_height - (value - min_y) / (max_y - min_y) * plot_height

    colors = {
        "fcfs": "#94a3b8",
        "static_batch": "#f59e0b",
        "continuous_batch": "#22c55e",
        "slo_aware": "#8b5cf6",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="20" fill="#08111f"/>',
        '<text x="100" y="35" fill="#f8fafc" font-family="system-ui" font-size="22" font-weight="700">Scheduler policy frontier</text>',
        '<text x="100" y="486" fill="#94a3b8" font-family="system-ui" font-size="14">Higher throughput →</text>',
        '<text x="22" y="235" fill="#94a3b8" font-family="system-ui" font-size="14" transform="rotate(-90 22 235)">Lower p95 latency →</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#334155"/>',
    ]
    for step in range(5):
        value = max_y * step / 4
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_width}" y2="{yy:.1f}" stroke="#172033"/>')
        parts.append(f'<text x="{left - 12}" y="{yy + 5:.1f}" text-anchor="end" fill="#64748b" font-family="system-ui" font-size="12">{value:.0f} ms</text>')
    for item in results:
        policy = item["policy"]
        xx = x(item["throughput_output_tokens_per_second"])
        yy = y(item["p95_latency_ms"])
        color = colors[policy]
        label = escape(policy.replace("_", " "))
        parts.extend([
            f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="10" fill="{color}" stroke="#f8fafc" stroke-width="2"/>',
            f'<text x="{xx + 15:.1f}" y="{yy - 8:.1f}" fill="#e2e8f0" font-family="system-ui" font-size="13" font-weight="600">{label}</text>',
            f'<text x="{xx + 15:.1f}" y="{yy + 10:.1f}" fill="#94a3b8" font-family="system-ui" font-size="11">{item["throughput_output_tokens_per_second"]:.0f} tok/s · {item["p95_latency_ms"]:.0f} ms</text>',
        ])
    parts.append('</svg>')
    return "\n".join(parts)
