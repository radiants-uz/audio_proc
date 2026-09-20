"""Диаграммы для отчёта: matplotlib → PNG bytes."""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _bar(labels: list[str], values: list[float], title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(6.5, max(1.8, 0.45 * len(labels) + 1.0)), dpi=150)
    ax.barh(labels[::-1], values[::-1], color="#3b6ea5")
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.set_title(title, fontsize=10)
    for i, v in enumerate(values[::-1]):
        ax.text(min(v + 1, 96), i, f"{v:.0f}%", va="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def speakers_chart(stats: dict, title: str) -> bytes:
    rows = stats.get("by_speaker", [])
    labels = [(r.get("name") or r["id"]) for r in rows]
    return _bar(labels, [r["share"] * 100 for r in rows], title)


def languages_chart(stats: dict, title: str) -> bytes:
    rows = stats.get("by_language", [])
    return _bar([r["language"] or "?" for r in rows], [r["share"] * 100 for r in rows], title)
